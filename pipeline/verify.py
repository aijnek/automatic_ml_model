"""VLM抽出品質のサンプル検証（step 4.5）。

正解ラベルなしで、特徴量ごとの抽出信頼性を測る:
- 自己一致率: 同一VLM・temperature>0 で n_repeats 回再抽出したときの一致率。
  低い＝プロンプトが曖昧で、同じ画像でも答えが揺れる
- 参照VLM一致率: 大型の参照VLMの抽出（temperature=0）との一致率。
  低い＝安定してブレないが間違っている可能性

train からの決定的サンプル画像に対し、新規/変更特徴量のみを対象に計測する
（差分方式・スキーマ収束後はコストゼロ）。結果は
results/iter{N}/extraction_quality.json に保存され、存在すればスキップ（再開対応）。
参照VLMが利用できない場合は cross_agreement=None で続行する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.config import Config
from pipeline.extract import (
    build_extraction_prompt,
    extract_one_image,
    features_to_extract,
    schema_to_json_schema,
)
from pipeline.designer import active_features
from pipeline.llm import vlm_chat

logger = logging.getLogger(__name__)

FLOAT_RELATIVE_TOLERANCE = 0.15  # float の一致判定: 相対誤差15%以内


def sample_images(
    train_filenames: list[str], sample_size: int, seed: int, iteration: int
) -> list[str]:
    """train から検証用サンプルを決定的に選ぶ（再実行しても同じ画像になる）。"""
    rng = np.random.default_rng(seed + iteration)
    n = min(sample_size, len(train_filenames))
    idx = rng.choice(len(train_filenames), size=n, replace=False)
    return [train_filenames[i] for i in sorted(idx)]


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def values_agree(feature: dict, a, b) -> bool | None:
    """型別の一致判定。両方NaNはペア除外（None）、片方NaNは不一致。"""
    if _is_nan(a) and _is_nan(b):
        return None
    if _is_nan(a) or _is_nan(b):
        return False
    if feature["type"] in ("binary", "categorical"):
        return a == b
    if feature["type"] == "scale_1_5":
        return abs(float(a) - float(b)) <= 1
    if feature["type"] == "float":
        a, b = float(a), float(b)
        return abs(a - b) <= FLOAT_RELATIVE_TOLERANCE * (abs(a) + abs(b)) / 2 + 1e-9
    raise ValueError(f"unknown feature type: {feature['type']}")


def consensus_value(feature: dict, values: list):
    """複数回抽出のコンセンサス値。float は中央値、他は最頻値。全NaNは NaN。"""
    valid = [v for v in values if not _is_nan(v)]
    if not valid:
        return np.nan
    if feature["type"] == "float":
        return float(np.median([float(v) for v in valid]))
    counts: dict = {}
    for v in valid:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def compute_feature_signals(
    feature: dict,
    repeats: list[dict[str, dict]],
    reference: dict[str, dict] | None,
) -> dict:
    """1特徴量分の信頼性シグナルを計算する。

    repeats: n_repeats 回分の {filename: {feature_name: value}}
    reference: 参照VLMの {filename: {feature_name: value}}（利用不可なら None）
    """
    name = feature["name"]
    filenames = list(repeats[0]) if repeats else []

    # 自己一致率: 画像ごとの全ペア一致率の平均
    per_image_rates = []
    n_values = 0
    n_nan = 0
    for fname in filenames:
        values = [r[fname].get(name, np.nan) for r in repeats]
        n_values += len(values)
        n_nan += sum(1 for v in values if _is_nan(v))
        pair_results = [
            values_agree(feature, values[i], values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
        valid_pairs = [p for p in pair_results if p is not None]
        if valid_pairs:
            per_image_rates.append(sum(valid_pairs) / len(valid_pairs))
    consistency = (
        float(np.mean(per_image_rates)) if per_image_rates and len(repeats) >= 2 else None
    )

    # 参照VLM一致率: コンセンサス値 vs 参照値
    cross_agreement = None
    if reference is not None:
        agreements = []
        for fname in filenames:
            consensus = consensus_value(feature, [r[fname].get(name, np.nan) for r in repeats])
            agree = values_agree(feature, consensus, reference.get(fname, {}).get(name, np.nan))
            if agree is not None:
                agreements.append(agree)
        if agreements:
            cross_agreement = float(np.mean(agreements))

    return {
        "consistency": consistency,
        "cross_agreement": cross_agreement,
        "sample_nan_rate": (n_nan / n_values) if n_values else None,
        "n_images": len(filenames),
        "n_repeats": len(repeats),
        "carried_forward": False,
    }


def classify_status(
    signals: dict | None, dist_stats: dict | None, cfg: Config
) -> tuple[str, list[str]]:
    """シグナルと全データ分布からルールベースの判定を返す。

    優先順: 抽出失敗 > 縮退 > 不安定 > 安定
    """
    reasons: list[str] = []
    if dist_stats and dist_stats.get("nan_rate") is not None:
        if dist_stats["nan_rate"] > cfg.verify_nan_rate_threshold:
            return "抽出失敗", [
                f"NaN率 {dist_stats['nan_rate']:.2f} > {cfg.verify_nan_rate_threshold}"
            ]
    if dist_stats and dist_stats.get("mode_fraction") is not None:
        if dist_stats["mode_fraction"] > cfg.verify_mode_fraction_threshold:
            return "縮退", [
                f"最頻値割合 {dist_stats['mode_fraction']:.2f} > {cfg.verify_mode_fraction_threshold}"
            ]
    if signals:
        consistency = signals.get("consistency")
        if consistency is not None and consistency < cfg.verify_consistency_threshold:
            reasons.append(
                f"自己一致率 {consistency:.2f} < {cfg.verify_consistency_threshold}"
            )
        cross = signals.get("cross_agreement")
        if cross is not None and cross < cfg.verify_agreement_threshold:
            reasons.append(
                f"参照VLM一致率 {cross:.2f} < {cfg.verify_agreement_threshold}"
            )
    if reasons:
        return "不安定", reasons
    return "安定", []


def _bind(vlm_fn: Callable[..., str], **extra) -> Callable[..., str]:
    """vlm_fn に temperature 等の追加 kwargs を固定して渡すラッパ。"""

    def fn(**kwargs):
        return vlm_fn(**kwargs, **extra)

    return fn


def _reference_available(
    targets: list[dict],
    image_path: Path,
    model: str,
    reference_fn: Callable[..., str],
) -> bool:
    """参照VLMが応答するか1コールで確認する（未pull等は即失敗する）。"""
    try:
        reference_fn(
            model=model,
            prompt=build_extraction_prompt(targets),
            image_path=image_path,
            format_schema=schema_to_json_schema(targets),
        )
        return True
    except Exception as e:
        logger.warning("参照VLM %s が利用できません（参照一致率は省略）: %s", model, e)
        return False


def run_verification(
    schema: dict,
    prev_schema: dict | None,
    train_filenames: list[str],
    images_dir: Path,
    cfg: Config,
    iteration: int,
    out_json: Path,
    prev_json: Path | None = None,
    vlm_fn: Callable[..., str] | None = None,
) -> dict:
    """新規/変更特徴量の抽出品質をサンプル検証し、結果を out_json に保存して返す。"""
    if out_json.exists():
        logger.info("iter %d: 既存の %s を使用（検証スキップ）", iteration, out_json.name)
        return json.loads(out_json.read_text())

    targets = features_to_extract(schema, prev_schema)
    feature_signals: dict[str, dict] = {}

    # kept 特徴量は前回の検証結果をキャリーフォワード（レポートが全特徴量をカバー）
    if prev_json is not None and prev_json.exists():
        prev_result = json.loads(prev_json.read_text())
        target_names = {f["name"] for f in targets}
        for f in active_features(schema):
            prev_signals = prev_result.get("features", {}).get(f["name"])
            if f["name"] not in target_names and prev_signals is not None:
                feature_signals[f["name"]] = {**prev_signals, "carried_forward": True}

    result = {
        "iteration": iteration,
        "sample_size": cfg.verify_sample_size,
        "n_repeats": cfg.verify_n_repeats,
        "reference_model": cfg.reference_vlm_model,
        "reference_available": None,
        "sampled": [],
        "features": feature_signals,
    }

    if not targets:
        logger.info("iter %d: 新規/変更特徴量なし → 検証スキップ", iteration)
        _save(result, out_json)
        return result

    sampled = sample_images(
        train_filenames, cfg.verify_sample_size, cfg.seed, iteration
    )
    result["sampled"] = sampled
    base_vlm = vlm_fn or vlm_chat
    primary_fn = _bind(base_vlm, temperature=cfg.verify_temperature)
    reference_fn = _bind(base_vlm, temperature=0.0, timeout=cfg.verify_vlm_timeout)

    # 主VLM: サンプル × n_repeats 回の再抽出
    logger.info(
        "iter %d: 抽出品質検証（%d特徴量 / サンプル%d枚 × %d回, %s）",
        iteration, len(targets), len(sampled), cfg.verify_n_repeats, cfg.vlm_model,
    )
    repeats: list[dict[str, dict]] = []
    for r in range(cfg.verify_n_repeats):
        run: dict[str, dict] = {}
        for fname in sampled:
            run[fname] = extract_one_image(
                targets, images_dir / fname, cfg.vlm_model, vlm_fn=primary_fn
            )
        repeats.append(run)
        logger.info("iter %d: 再抽出 %d/%d 完了", iteration, r + 1, cfg.verify_n_repeats)

    # 参照VLM: サンプル × 1回（利用不可・途中失敗でも続行）
    reference: dict[str, dict] | None = None
    if _reference_available(
        targets, images_dir / sampled[0], cfg.reference_vlm_model, reference_fn
    ):
        result["reference_available"] = True
        logger.info("iter %d: 参照VLM (%s) で抽出中...", iteration, cfg.reference_vlm_model)
        try:
            reference = {
                fname: extract_one_image(
                    targets, images_dir / fname, cfg.reference_vlm_model,
                    vlm_fn=reference_fn,
                )
                for fname in sampled
            }
        except Exception as e:  # 自律ループ優先で握りつぶす
            logger.warning("参照VLM抽出が失敗しました（参照一致率は省略）: %s", e)
            reference = None
            result["reference_available"] = False
    else:
        result["reference_available"] = False

    for f in targets:
        feature_signals[f["name"]] = compute_feature_signals(f, repeats, reference)

    _save(result, out_json)
    return result


def _save(result: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
