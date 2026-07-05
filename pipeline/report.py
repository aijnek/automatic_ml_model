"""イテレーション診断レポートの生成（designer へのフィードバック材料）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.config import Config
from pipeline.verify import classify_status


def _fmt_value(v) -> str:
    if v is None:
        return "NaN"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _fmt_rate(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def compute_distribution_stats(
    features_df: pd.DataFrame, features: list[dict]
) -> dict[str, dict]:
    """全データの特徴量CSVから NaN率・最頻値割合・ユニーク数を計算する。"""
    stats: dict[str, dict] = {}
    for f in features:
        name = f["name"]
        if name not in features_df.columns:
            continue
        col = features_df[name]
        valid = col.dropna()
        if f["type"] == "float":
            valid = valid.round(3)  # 微小差を同一視して縮退を検出
        stats[name] = {
            "nan_rate": float(col.isna().mean()) if len(col) else None,
            "mode_fraction": (
                float(valid.value_counts(normalize=True).iloc[0]) if len(valid) else None
            ),
            "n_unique": int(valid.nunique()),
        }
    return stats


def build_quality_records(
    cfg: Config,
    extraction_quality: dict,
    dist_stats: dict[str, dict],
    features: list[dict],
) -> dict[str, dict]:
    """検証シグナル＋全データ分布を特徴量ごとにマージし、status を判定する。"""
    records: dict[str, dict] = {}
    signals_by_name = extraction_quality.get("features", {})
    for f in features:
        name = f["name"]
        signals = signals_by_name.get(name)
        dist = dist_stats.get(name)
        status, reasons = classify_status(signals, dist, cfg)
        records[name] = {
            "status": status,
            "reasons": reasons,
            "consistency": (signals or {}).get("consistency"),
            "cross_agreement": (signals or {}).get("cross_agreement"),
            "carried_forward": (signals or {}).get("carried_forward", False),
            "nan_rate": (dist or {}).get("nan_rate"),
            "mode_fraction": (dist or {}).get("mode_fraction"),
            "n_unique": (dist or {}).get("n_unique"),
        }
    return records


def _quality_section(quality_records: dict[str, dict], reference_model: str) -> list[str]:
    lines = [
        "",
        "## 抽出信頼性（VLMメタデータ抽出の品質検証）",
        "",
        f"trainサンプル画像の複数回再抽出による自己一致率と、参照VLM（{reference_model}）"
        "との一致率:",
        "",
        "| 特徴量 | 判定 | 自己一致率 | 参照VLM一致率 | NaN率(全データ) | 最頻値割合 |",
        "|---|---|---|---|---|---|",
    ]
    for name, rec in quality_records.items():
        status = rec["status"]
        if rec["reasons"]:
            status += " ← " + "、".join(rec["reasons"])
        lines.append(
            f"| {name} | {status} | {_fmt_rate(rec['consistency'])} "
            f"| {_fmt_rate(rec['cross_agreement'])} | {_fmt_rate(rec['nan_rate'])} "
            f"| {_fmt_rate(rec['mode_fraction'])} |"
        )
    lines += [
        "",
        "- 判定の意味: 安定=抽出は信頼できる / 不安定=同じ画像でも値が揺れる"
        "（プロンプトが曖昧） / 縮退=ほぼ全画像で同じ値（情報量なし） / "
        "抽出失敗=VLMが答えられずNaNになる",
        "- 「重要度が低い」は必ずしも観点が悪いとは限らない。不安定な特徴量は"
        "抽出ノイズで重要度が過小評価されている可能性がある。",
    ]
    return lines


def build_report_md(
    cfg: Config,
    iteration: int,
    result: dict,
    quality_records: dict[str, dict] | None = None,
    reference_model: str = "",
) -> str:
    """人間可読 & designer 入力用のマークダウンレポート。"""
    lines = [
        f"# イテレーション {iteration} 診断レポート",
        "",
        f"- val {result['metric_name']}: **{result['val_score']:.4f}**"
        f"（合格ライン: {cfg.threshold}）",
        f"- train/val サンプル数: {result['n_train']} / {result['n_val']}",
        "",
    ]

    m = result["val_metrics"]
    if cfg.is_classification:
        lines.append("## クラス別 F1")
        for cls, f in m["per_class_f1"].items():
            lines.append(f"- {cls}: {f:.3f}")
        lines += ["", "## 混同行列（行=正解, 列=予測）", ""]
        labels = m["confusion_matrix"]["labels"]
        lines.append("| | " + " | ".join(f"pred:{c}" for c in labels) + " |")
        lines.append("|" + "---|" * (len(labels) + 1))
        for cls, row in zip(labels, m["confusion_matrix"]["matrix"]):
            lines.append(f"| true:{cls} | " + " | ".join(str(x) for x in row) + " |")
    else:
        lines += [
            "## 回帰指標",
            f"- R²: {m['r2']:.4f}",
            f"- MAE: {m['mae']:.4f}",
            f"- RMSE: {m['rmse']:.4f}",
            f"- 残差: mean={m['residual_stats']['mean']:.3f}, "
            f"std={m['residual_stats']['std']:.3f}, "
            f"max|e|={m['residual_stats']['max_abs']:.3f}",
        ]

    lines += ["", "## 特徴量重要度（gain, 正規化済み・降順）"]
    for name, imp in result["feature_importances"].items():
        flag = " ← 寄与ほぼゼロ" if imp < 0.02 else ""
        lines.append(f"- {name}: {imp:.3f}{flag}")

    if quality_records:
        lines += _quality_section(quality_records, reference_model)

    worst = result["worst_val_samples"]
    if worst:
        title = "誤分類サンプル" if cfg.is_classification else "高誤差サンプル"
        lines += ["", f"## val {title}（最大{len(worst)}件、特徴量値付き）"]
        for rec in worst:
            feat_str = ", ".join(
                f"{k}={_fmt_value(v)}" for k, v in rec["features"].items()
            )
            lines.append(
                f"- {rec['filename']}: 正解={_fmt_value(rec['true'])}, "
                f"予測={_fmt_value(rec['predicted'])} | {feat_str}"
            )

    return "\n".join(lines) + "\n"


def save_iteration_report(
    cfg: Config,
    iteration: int,
    result: dict,
    out_dir: Path,
    extraction_quality: dict | None = None,
    features_df: pd.DataFrame | None = None,
    schema: dict | None = None,
) -> str:
    """report.json / report.md を保存し、markdown を返す。

    extraction_quality（verify.run_verification の結果）と features_df / schema が
    揃っていれば、抽出信頼性セクションをレポートに含める。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if not k.startswith("_")}

    quality_records = None
    reference_model = ""
    if extraction_quality is not None and features_df is not None and schema is not None:
        from pipeline.designer import active_features  # 循環import回避

        features = active_features(schema)
        dist_stats = compute_distribution_stats(features_df, features)
        quality_records = build_quality_records(
            cfg, extraction_quality, dist_stats, features
        )
        reference_model = extraction_quality.get("reference_model", "")
        serializable["extraction_quality"] = quality_records

    (out_dir / "report.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2)
    )
    md = build_report_md(cfg, iteration, result, quality_records, reference_model)
    (out_dir / "report.md").write_text(md)
    return md


def build_final_report_md(
    cfg: Config,
    best_iteration: int,
    history: list[dict],
    test_result: dict,
    schema: dict,
    selection: dict | None = None,
) -> str:
    lines = [
        "# 最終レポート",
        "",
        f"- タスク: {cfg.task_type}（{cfg.description}）",
        f"- 採用イテレーション: {best_iteration}",
        f"- **test {cfg.metric_name}: {test_result['test_score']:.4f}**"
        f"（testサンプル数: {test_result['n_test']}）",
        "",
    ]
    m = test_result["test_metrics"]
    if cfg.is_classification:
        lines.append("## test クラス別 F1")
        for cls, f in m["per_class_f1"].items():
            lines.append(f"- {cls}: {f:.3f}")
        lines += ["", "## test 混同行列（行=正解, 列=予測）", ""]
        labels = m["confusion_matrix"]["labels"]
        lines.append("| | " + " | ".join(f"pred:{c}" for c in labels) + " |")
        lines.append("|" + "---|" * (len(labels) + 1))
        for cls, row in zip(labels, m["confusion_matrix"]["matrix"]):
            lines.append(f"| true:{cls} | " + " | ".join(str(x) for x in row) + " |")
    else:
        lines += [
            "## test 回帰指標",
            f"- R²: {m['r2']:.4f}",
            f"- MAE: {m['mae']:.4f}",
            f"- RMSE: {m['rmse']:.4f}",
        ]

    lines += ["", "## イテレーションごとの val スコア推移"]
    for h in history:
        marker = " ← 採用" if h["iteration"] == best_iteration else ""
        lines.append(
            f"- iter {h['iteration']}: val {cfg.metric_name} = {h['val_score']:.4f}"
            f"（特徴量 {h['n_features']}個）{marker}"
        )

    if selection is not None:
        use_cv = selection.get("cv_enabled", False)
        score_label = f"CV({selection['cv_folds']}分割) {cfg.metric_name}" if use_cv else f"val {cfg.metric_name}"
        baseline_score = selection["baseline_cv_score"] if use_cv else selection["baseline_val_score"]
        final_score = selection["final_cv_score"] if use_cv else selection["final_val_score"]
        lines += [
            "",
            "## 特徴量選択（後方消去）",
            "",
            f"- 採否判定に使った{score_label}: {baseline_score:.4f} → {final_score:.4f}"
            f"（許容低下 {selection['max_score_drop']}）",
        ]
        if use_cv:
            lines.append(
                f"- 参考: val単一分割 {cfg.metric_name}: "
                f"{selection['baseline_val_score']:.4f} → {selection['final_val_score']:.4f}"
            )
        lines += [
            f"- 特徴量数: {selection['n_features_before']} → "
            f"{selection['n_features_after']}",
            "- 除去: "
            + (", ".join(selection["removed"]) if selection["removed"] else "なし"),
        ]
        if selection["rounds"]:
            lines += [
                "",
                f"| ラウンド | 除去候補 | 重要度 | {score_label} | 低下 | 判定 |",
                "|---|---|---|---|---|---|",
            ]
            for i, r in enumerate(selection["rounds"], start=1):
                lines.append(
                    f"| {i} | {r['feature']} | {r['importance']:.4f} "
                    f"| {r['score']:.4f} | {r['score_drop']:+.4f} "
                    f"| {'採用' if r['accepted'] else '棄却'} |"
                )

    lines += ["", "## 最終特徴量スキーマ"]
    from pipeline.designer import active_features  # 循環import回避

    for f in active_features(schema):
        lines.append(f"- **{f['name']}** ({f['type']}): {f['prompt']}")

    return "\n".join(lines) + "\n"
