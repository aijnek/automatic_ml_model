"""VLM（ollama）による画像メタデータ抽出（step 5）。

- 画像1枚につき1コールで、変更のあった特徴量だけをまとめて質問する
- ollama の structured outputs（format=JSON Schema）で型を保証
- 定義が前回スキーマと同一の特徴量は features_v{N-1}.csv から値を流用（差分抽出）
- 1枚処理するごとに CSV へ追記保存し、中断→再開できる
- リトライしても失敗した画像・特徴量は NaN（LightGBM は NaN を扱える）
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pipeline.designer import active_features, extract_json
from pipeline.llm import vlm_chat

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_IMAGE = 1  # バッチ抽出の再試行。失敗分は単項目フォールバックで拾う


def _definition(feature: dict) -> tuple:
    """差分抽出の同一性判定に使う特徴量定義のキー。"""
    return (
        feature["type"],
        tuple(feature.get("choices") or ()),
        feature["prompt"].strip(),
    )


def features_to_extract(schema: dict, prev_schema: dict | None) -> list[dict]:
    """VLM に問い合わせが必要な特徴量（新規 or 定義変更）を返す。"""
    if prev_schema is None:
        return active_features(schema)
    prev = {f["name"]: _definition(f) for f in active_features(prev_schema)}
    return [
        f
        for f in active_features(schema)
        if prev.get(f["name"]) != _definition(f)
    ]


def schema_to_json_schema(features: list[dict]) -> dict:
    """特徴量リストから ollama structured outputs 用の JSON Schema を作る。"""
    props: dict[str, dict] = {}
    for f in features:
        if f["type"] == "scale_1_5":
            props[f["name"]] = {"type": "integer", "minimum": 1, "maximum": 5}
        elif f["type"] == "binary":
            props[f["name"]] = {"type": "boolean"}
        elif f["type"] == "categorical":
            props[f["name"]] = {"type": "string", "enum": list(f["choices"])}
        elif f["type"] == "float":
            props[f["name"]] = {"type": "number"}
        else:
            raise ValueError(f"unknown feature type: {f['type']}")
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
    }


def build_extraction_prompt(features: list[dict]) -> str:
    lines = [
        "Look at this image carefully and answer the following questions.",
        "Respond in JSON with exactly these keys.",
        "",
    ]
    for f in features:
        lines.append(f"- {f['name']}: {f['prompt']}")
    return "\n".join(lines)


def _repair_json(text: str) -> str:
    """VLMが出しがちな不正JSONを修復する（例: `"key": yes` の未クオート値）。"""
    return re.sub(
        r':\s*(yes|no|Yes|No|N/A|null)\s*([,}\]])',
        lambda m: f': "{m.group(1)}"{m.group(2)}',
        text,
    )


def parse_vlm_json(raw: str) -> dict:
    """VLM出力を寛容にパースする。ollamaのformat強制が効かないケースがあるため必須。"""
    for candidate in (raw, _repair_json(raw)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return extract_json(_repair_json(raw))  # 前後の文章を除いて部分抽出


def coerce_value(feature: dict, value: object) -> object:
    """VLM出力の値を学習用の型に正規化する。不正値は NaN。"""
    try:
        if feature["type"] == "binary":
            if isinstance(value, str):
                value = value.strip().lower() in ("true", "yes", "1")
            return int(bool(value))
        if feature["type"] == "scale_1_5":
            v = int(round(float(value)))
            return v if 1 <= v <= 5 else np.nan
        if feature["type"] == "float":
            return float(value)
        if feature["type"] == "categorical":
            v = str(value).strip()
            return v if v in feature["choices"] else np.nan
    except (TypeError, ValueError):
        return np.nan
    raise ValueError(f"unknown feature type: {feature['type']}")


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def extract_one_image(
    features: list[dict],
    image_path: Path,
    model: str,
    vlm_fn: Callable[..., str] | None = None,
) -> dict[str, object]:
    """1枚の画像から指定特徴量を抽出する。

    全項目まとめてのバッチ抽出を試み、取れなかった項目だけ1項目ずつ
    フォールバック抽出する。それでも失敗した特徴量は NaN。
    """
    if vlm_fn is None:
        vlm_fn = vlm_chat
    values: dict[str, object] = {}

    for attempt in range(MAX_RETRIES_PER_IMAGE + 1):
        try:
            raw = vlm_fn(
                model=model,
                prompt=build_extraction_prompt(features),
                image_path=image_path,
                format_schema=schema_to_json_schema(features),
            )
            parsed = parse_vlm_json(raw)
            values = {
                f["name"]: coerce_value(f, parsed[f["name"]])
                for f in features
                if f["name"] in parsed
            }
            break
        except Exception as e:  # ollama接続断・不正JSONなど。自律ループ優先で握りつぶす
            logger.warning(
                "バッチ抽出失敗 (%s, attempt %d/%d): %s",
                image_path.name,
                attempt + 1,
                MAX_RETRIES_PER_IMAGE + 1,
                e,
            )

    missing = [f for f in features if _is_missing(values.get(f["name"]))]
    for f in missing:
        try:
            raw = vlm_fn(
                model=model,
                prompt=build_extraction_prompt([f]),
                image_path=image_path,
                format_schema=schema_to_json_schema([f]),
            )
            values[f["name"]] = coerce_value(f, parse_vlm_json(raw).get(f["name"]))
        except Exception as e:
            logger.warning(
                "単項目抽出も失敗 (%s / %s): %s", image_path.name, f["name"], e
            )
            values[f["name"]] = np.nan

    return {f["name"]: values.get(f["name"], np.nan) for f in features}


def run_extraction(
    schema: dict,
    filenames: list[str],
    images_dir: Path,
    out_csv: Path,
    model: str,
    prev_schema: dict | None = None,
    prev_csv: Path | None = None,
    vlm_fn: Callable[..., str] | None = None,
) -> pd.DataFrame:
    """全画像の特徴量CSV（filename + 特徴量列）を作成して返す。

    - prev_csv があれば定義の変わっていない特徴量の値を流用
    - out_csv が途中まで存在すれば処理済み画像をスキップ（再開）
    """
    feats = active_features(schema)
    feat_names = [f["name"] for f in feats]
    columns = ["filename"] + feat_names

    to_extract = features_to_extract(schema, prev_schema)
    to_extract_names = [f["name"] for f in to_extract]
    reuse_names = [n for n in feat_names if n not in to_extract_names]

    cached = pd.DataFrame()
    if reuse_names and prev_csv is not None and prev_csv.exists():
        cached = pd.read_csv(prev_csv).set_index("filename")
        missing = [n for n in reuse_names if n not in cached.columns]
        if missing:  # キャッシュに無い列は抽出に回す（安全側）
            to_extract = to_extract + [f for f in feats if f["name"] in missing]
            to_extract_names += missing
            reuse_names = [n for n in reuse_names if n not in missing]

    done: set[str] = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        if list(existing.columns) == columns:
            # 全特徴量NaNの行は抽出失敗の記録なので、再開時に再抽出する
            valid = ~existing[feat_names].isna().all(axis=1)
            done = set(existing.loc[valid, "filename"].astype(str))
            n_retry = int((~valid).sum())
            logger.info(
                "既存の %s から再開: %d 枚処理済み%s",
                out_csv.name,
                len(done),
                f"（全NaNの{n_retry}枚は再抽出）" if n_retry else "",
            )
        else:  # スキーマが変わっていたら作り直し
            out_csv.unlink()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(filenames)
    for i, filename in enumerate(filenames):
        if filename in done:
            continue
        row: dict[str, object] = {"filename": filename}
        for name in reuse_names:
            row[name] = cached[name].get(filename, np.nan) if not cached.empty else np.nan
        if to_extract:
            row.update(
                extract_one_image(to_extract, images_dir / filename, model, vlm_fn)
            )
        pd.DataFrame([row], columns=columns).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )
        done.add(filename)
        if (i + 1) % 10 == 0 or i + 1 == n_total:
            logger.info("抽出進捗: %d / %d", len(done), n_total)

    # 再抽出でファイル名が重複した場合は最新行を採用
    df = pd.read_csv(out_csv).drop_duplicates("filename", keep="last")
    return df[df["filename"].isin(filenames)].reset_index(drop=True)
