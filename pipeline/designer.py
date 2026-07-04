"""LLM による特徴量スキーマの設計・改訂（step 4）。

スキーマ形式 (schemas/metadata_v{N}.json):
{
  "version": N,
  "features": [
    {"name": "...", "type": "scale_1_5|binary|categorical|float",
     "choices": [...],           # categorical のみ
     "prompt": "...",            # VLM への質問文（英語）
     "action": "new|modified|kept|removed",
     "rationale": "..."}
  ]
}

action は designer の宣言だが、差分抽出の判定は extract.py が
定義（type/choices/prompt）の同一性で行うため、action の誤りは安全側に倒れる。
"""

from __future__ import annotations

import json
import re
from typing import Callable

from pipeline.config import Config
from pipeline.llm import run_designer_llm

VALID_TYPES = {"scale_1_5", "binary", "categorical", "float"}
MIN_FEATURES = 3
MAX_FEATURES = 15
MAX_RETRIES = 3

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SchemaValidationError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """LLM出力から最初のJSONオブジェクトを抽出してパースする。

    ```json フェンス優先、なければ最初の '{' から括弧の対応でスキャン。
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    if start == -1:
        raise SchemaValidationError("出力にJSONオブジェクトが見つかりません")
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise SchemaValidationError("JSONオブジェクトが閉じていません")


def active_features(schema: dict) -> list[dict]:
    return [f for f in schema["features"] if f.get("action") != "removed"]


def validate_schema(schema: dict, check_count: bool = True) -> None:
    feats = schema.get("features")
    if not isinstance(feats, list):
        raise SchemaValidationError("'features' がリストではありません")
    names = set()
    n_active = 0
    for f in feats:
        name = f.get("name", "")
        if not _NAME_RE.match(name):
            raise SchemaValidationError(
                f"特徴量名は snake_case にしてください: {name!r}"
            )
        if name in names:
            raise SchemaValidationError(f"特徴量名が重複しています: {name}")
        names.add(name)
        if f.get("type") not in VALID_TYPES:
            raise SchemaValidationError(
                f"{name}: type は {sorted(VALID_TYPES)} のいずれかにしてください: {f.get('type')!r}"
            )
        if f.get("type") == "categorical":
            choices = f.get("choices")
            if not isinstance(choices, list) or len(choices) < 2:
                raise SchemaValidationError(
                    f"{name}: categorical には choices（2つ以上）が必要です"
                )
        if not isinstance(f.get("prompt"), str) or not f["prompt"].strip():
            raise SchemaValidationError(f"{name}: prompt が空です")
        if f.get("action", "new") != "removed":
            n_active += 1
    if not check_count:
        return
    if n_active < MIN_FEATURES:
        raise SchemaValidationError(
            f"有効な特徴量が少なすぎます ({n_active}個)。{MIN_FEATURES}個以上にしてください。"
        )
    if n_active > MAX_FEATURES:
        raise SchemaValidationError(
            f"有効な特徴量が多すぎます ({n_active}個)。{MAX_FEATURES}個以下にしてください。"
        )


def merge_schema(prev_schema: dict, proposed: dict, version: int) -> dict:
    """designer の提案を前回スキーマにマージする。

    - new / modified: 提案の定義を採用
    - kept: 前回の定義をそのまま使う（プロンプトの意図しない揺れで再抽出しないため）
    - removed: 除外（記録として action=removed で残す）
    - 提案に現れなかった前回の特徴量: kept 扱い
    """
    prev_by_name = {f["name"]: f for f in active_features(prev_schema)}
    merged: dict[str, dict] = {}
    for f in proposed["features"]:
        name = f["name"]
        action = f.get("action", "new")
        if action == "kept" and name in prev_by_name:
            entry = {**prev_by_name[name], "action": "kept"}
        elif action == "removed":
            entry = {**f, "action": "removed"}
        else:
            action = "modified" if name in prev_by_name else "new"
            entry = {**f, "action": action}
        merged[name] = entry
    for name, f in prev_by_name.items():
        if name not in merged:
            merged[name] = {**f, "action": "kept"}
    return {"version": version, "features": list(merged.values())}


def _task_description(cfg: Config) -> str:
    lines = [
        f"タスク種別: {'分類' if cfg.is_classification else '回帰'}",
        f"タスク説明: {cfg.description}",
    ]
    if cfg.is_classification:
        lines.append(f"クラス: {', '.join(cfg.classes)}")
    else:
        unit = f" (単位: {cfg.target_unit})" if cfg.target_unit else ""
        lines.append(f"ターゲット範囲: {cfg.target_min} 〜 {cfg.target_max}{unit}")
    return "\n".join(lines)


_FORMAT_SPEC = """\
出力は以下の形式のJSONオブジェクト1個のみ。説明文やマークダウンは一切不要。

{
  "features": [
    {
      "name": "snake_caseの特徴量名",
      "type": "scale_1_5" | "binary" | "categorical" | "float",
      "choices": ["categorical の場合のみ、選択肢のリスト"],
      "prompt": "VLMへの質問文。英語で、その画像1枚を見て答えられる具体的な質問にする。scale_1_5なら1と5のアンカー定義を含める。",
      "action": "new" | "modified" | "kept" | "removed",
      "rationale": "この特徴量がターゲット予測に効く理由（日本語可）"
    }
  ]
}

制約:
- 有効な特徴量（removed以外）は5〜12個
- 各特徴量は小型VLM（9B）が画像1枚から安定して答えられる粒度にする
- 主観的すぎる質問や、複数画像の比較が必要な質問は避ける
- prompt は、同じ画像に対して常に同じ答えが返る客観的な質問にする（曖昧な形容の度合いを聞かない）
"""


def build_initial_prompt(cfg: Config) -> str:
    return f"""あなたは画像認識タスクの特徴量設計者です。
画像からマルチモーダルLLM（VLM）でメタデータ（特徴量）を抽出し、
それをLightGBMに入力して以下のタスクを解きます。

{_task_description(cfg)}

このタスクのターゲットを予測するのに有効な特徴量を設計してください。

{_FORMAT_SPEC}"""


def build_revision_prompt(cfg: Config, prev_schema: dict, prev_report_md: str) -> str:
    schema_json = json.dumps(
        {"features": active_features(prev_schema)}, ensure_ascii=False, indent=1
    )
    return f"""あなたは画像認識タスクの特徴量設計者です。
画像からVLMでメタデータ（特徴量）を抽出し、LightGBMで以下のタスクを解いています。

{_task_description(cfg)}

## 現在の特徴量スキーマ
{schema_json}

## 前回イテレーションの診断レポート
{prev_report_md}

診断レポートを分析し、スコアを改善するために特徴量スキーマを改訂してください。
- 重要度が低い・予測に寄与していない特徴量は removed または modified（定義の具体化）
- 誤分類/高誤差サンプルの傾向から、不足している情報を補う特徴量を new で追加
- 問題ない特徴量は kept とし、features リストに必ず含めること
- 1回の改訂での変更（new+modified+removed）は1〜4個程度に絞ること
- レポートに「抽出信頼性」セクションがある場合は、重要度と組み合わせて次の方針で判断すること:
  - 判定が「不安定」で、タスクに有効そうな観点 → modified: VLMへのpromptを書き直す。
    書き直しの手法: 判定基準を具体的・観察可能な条件で定義する / 複雑な質問を単純な
    質問に分解する / categorical は choices を減らし境界を明確にする /
    scale_1_5 は 1〜5 の各段階（少なくとも1・3・5）のアンカーを明示する /
    主観的な度合いの質問は binary（有無の判定）への変更を検討する
  - 「不安定」かつ重要度も低い → removed
  - 「安定」だが重要度がほぼゼロ → removed（抽出は正しく行われているため、
    promptの修正では改善しない。観点自体が予測に寄与していない）
  - 「縮退」（ほぼ全画像で同じ値）→ 画像間の差を捉えられる定義に modified、または removed
  - 「抽出失敗」（NaN率が高い）→ VLMが答えられない質問。より単純で直接的な質問に modified

{_FORMAT_SPEC}"""


def design_schema(
    cfg: Config,
    version: int,
    prev_schema: dict | None = None,
    prev_report_md: str | None = None,
    llm_fn: Callable[[str], str] | None = None,
) -> dict:
    """特徴量スキーマを設計（v1）または改訂（v2以降）して返す。

    llm_fn はテスト用の差し替え口。既定では run_designer_llm を使う。
    """
    if llm_fn is None:
        llm_fn = lambda p: run_designer_llm(p, cfg.designer_command)  # noqa: E731

    if prev_schema is None:
        prompt = build_initial_prompt(cfg)
    else:
        prompt = build_revision_prompt(cfg, prev_schema, prev_report_md or "(レポートなし)")

    last_error = ""
    for attempt in range(MAX_RETRIES):
        current_prompt = prompt
        if last_error:
            current_prompt = (
                f"{prompt}\n\n前回のあなたの出力は次のエラーで拒否されました。"
                f"修正して再出力してください:\n{last_error}"
            )
        output = llm_fn(current_prompt)
        try:
            proposed = extract_json(output)
            # 改訂時の提案は差分なので、特徴量数の検証はマージ後にのみ行う
            validate_schema(proposed, check_count=prev_schema is None)
            if prev_schema is not None:
                schema = merge_schema(prev_schema, proposed, version)
                validate_schema(schema)
            else:
                schema = {"version": version, "features": proposed["features"]}
        except (SchemaValidationError, json.JSONDecodeError) as e:
            last_error = str(e)
            continue
        return schema
    raise SchemaValidationError(
        f"designer が {MAX_RETRIES} 回とも不正な出力を返しました。最後のエラー: {last_error}"
    )
