"""アノテーションバッチCSVを検証・マージして data/annotations.csv を生成する。

image-ml-agent スキルの Phase 4（全量アノテーション）で、サブエージェントが
data/annotation/batches/batch_*.csv に書いた行
（filename,label,confidence,reason）を1つに統合する。

使い方（プロジェクトルートから）:
    uv run python .claude/skills/image-ml-agent/scripts/merge_annotations.py
    uv run python .claude/skills/image-ml-agent/scripts/merge_annotations.py --dry-run

処理:
    1. config.yaml から許容ラベル（分類: classes / 回帰: target_range）を読む
    2. バッチCSVを連結し、filename 重複は後勝ちで排除
    3. DISCARD 行を除外（件数と理由の内訳を表示）
    4. 不正ラベル・data/images/ に実在しないファイルがあればエラーで終了
    5. data/annotations.csv（filename,label）を書き出し、ラベル分布を表示
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists():
            return d
    raise SystemExit("pyproject.toml が見つかりません。プロジェクトルートから実行してください")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import ANNOTATIONS_CSV, IMAGES_DIR, load_config  # noqa: E402

BATCHES_DIR = PROJECT_ROOT / "data" / "annotation" / "batches"
DISCARD = "DISCARD"


def read_batches(batches_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """全バッチを読み、filename -> 行（後勝ち）と発生順のエラーを返す。"""
    rows: dict[str, dict] = {}
    errors: list[str] = []
    files = sorted(batches_dir.glob("batch_*.csv"))
    if not files:
        raise SystemExit(f"バッチCSVがありません: {batches_dir}/batch_*.csv")
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = {"filename", "label"} - set(reader.fieldnames or [])
            if missing:
                errors.append(f"{path.name}: 列が不足 {sorted(missing)}")
                continue
            for i, row in enumerate(reader, start=2):
                filename = (row.get("filename") or "").strip()
                label = (row.get("label") or "").strip()
                if not filename or not label:
                    errors.append(f"{path.name}:{i}: filename または label が空")
                    continue
                rows[filename] = {
                    "filename": filename,
                    "label": label,
                    "reason": (row.get("reason") or "").strip(),
                    "batch": path.name,
                }
    return rows, errors


def validate_label(label: str, cfg) -> str | None:
    """ラベルを検証し、エラーメッセージ（問題なければ None）を返す。"""
    if cfg.is_classification:
        if label not in cfg.classes:
            return f"classes {cfg.classes} にないラベル: {label!r}"
        return None
    try:
        value = float(label)
    except ValueError:
        return f"数値に変換できないラベル: {label!r}"
    if cfg.target_min is not None and value < cfg.target_min:
        return f"target_range 未満の値: {value}"
    if cfg.target_max is not None and value > cfg.target_max:
        return f"target_range 超過の値: {value}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="annotations.csv を書かずに検証と集計だけ行う",
    )
    args = parser.parse_args()

    cfg = load_config()
    rows, errors = read_batches(BATCHES_DIR)

    kept: list[dict] = []
    discard_reasons: Counter[str] = Counter()
    for row in rows.values():
        if row["label"] == DISCARD:
            discard_reasons[row["reason"] or "(理由なし)"] += 1
            continue
        err = validate_label(row["label"], cfg)
        if err:
            errors.append(f"{row['batch']}: {row['filename']}: {err}")
            continue
        if not (IMAGES_DIR / row["filename"]).exists():
            errors.append(f"{row['batch']}: 画像が存在しません: {row['filename']}")
            continue
        kept.append(row)

    n_discard = sum(discard_reasons.values())
    print(f"バッチ行数: {len(rows)}（採用 {len(kept)} / DISCARD {n_discard} / 不正 {len(errors)}）")
    if discard_reasons:
        print("DISCARD 理由の内訳:")
        for reason, n in discard_reasons.most_common():
            print(f"  {n:4d}  {reason}")
    if errors:
        print("エラー:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit("不正な行があります。バッチを修正して再実行してください")
    if not kept:
        raise SystemExit("採用された行が0件です")

    print("ラベル分布:")
    for label, n in Counter(r["label"] for r in kept).most_common():
        print(f"  {n:4d}  {label}")

    if args.dry_run:
        print("(dry-run: annotations.csv は書き出していません)")
        return

    kept.sort(key=lambda r: r["filename"])
    with open(ANNOTATIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    print(f"{ANNOTATIONS_CSV} に {len(kept)} 行を書き出しました")


if __name__ == "__main__":
    main()
