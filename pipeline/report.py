"""イテレーション診断レポートの生成（designer へのフィードバック材料）。"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import Config


def _fmt_value(v) -> str:
    if v is None:
        return "NaN"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def build_report_md(cfg: Config, iteration: int, result: dict) -> str:
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
    cfg: Config, iteration: int, result: dict, out_dir: Path
) -> str:
    """report.json / report.md を保存し、markdown を返す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if not k.startswith("_")}
    (out_dir / "report.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2)
    )
    md = build_report_md(cfg, iteration, result)
    (out_dir / "report.md").write_text(md)
    return md


def build_final_report_md(
    cfg: Config,
    best_iteration: int,
    history: list[dict],
    test_result: dict,
    schema: dict,
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

    lines += ["", "## 最終特徴量スキーマ"]
    from pipeline.designer import active_features  # 循環import回避

    for f in active_features(schema):
        lines.append(f"- **{f['name']}** ({f['type']}): {f['prompt']}")

    return "\n".join(lines) + "\n"
