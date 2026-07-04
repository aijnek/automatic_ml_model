"""合格モデルに対する特徴量選択（重要度ベースの後方消去, step 7.5）。

val 合格後・test 評価前に実行する。gain 重要度が最小の特徴量を1つずつ除去して
再学習し、baseline（合格モデルの val スコア）からの低下が
cfg.select_max_score_drop 以内なら採用、超えたら棄却して終了する。

- test split は絶対に参照しない（最終評価は run_loop の step 8 の1回のみ）
- LLM 呼び出しなし・seed 固定で決定的なので、中断時は再実行でやり直すだけでよい
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import pandas as pd

from pipeline.config import Config
from pipeline.designer import active_features
from pipeline.train import train_and_evaluate

logger = logging.getLogger(__name__)


def select_features(
    cfg: Config,
    schema: dict,
    features_df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    baseline_result: dict,
    out_json: Path | None = None,
    schema_path: Path | None = None,
    model_path: Path | None = None,
) -> dict:
    """後方消去を実行し、選択後スキーマ・モデル・履歴を含む dict を返す。

    splits は train/val のみ参照する。baseline_result は合格モデルの
    train_and_evaluate 結果（feature_importances と _model を含む）。
    """
    baseline = baseline_result["val_score"]
    current_schema = deepcopy(schema)
    current_result = baseline_result
    n_before = len(active_features(current_schema))
    rounds: list[dict] = []
    removed: list[str] = []

    logger.info(
        "特徴量選択開始: %d特徴量, baseline val %s = %.4f（許容低下 %.3f, 最小特徴量数 %d）",
        n_before,
        cfg.metric_name,
        baseline,
        cfg.select_max_score_drop,
        cfg.select_min_features,
    )

    while len(active_features(current_schema)) > cfg.select_min_features:
        importances = current_result["feature_importances"]
        target = min(
            active_features(current_schema),
            key=lambda f: importances.get(f["name"], 0.0),
        )
        importance = importances.get(target["name"], 0.0)

        candidate = deepcopy(current_schema)
        for f in candidate["features"]:
            if f["name"] == target["name"]:
                f["action"] = "removed"

        cand_result = train_and_evaluate(cfg, candidate, features_df, splits)
        drop = baseline - cand_result["val_score"]
        # 許容低下のみで判定する（cfg.threshold を下回っても baseline との差が
        # select_max_score_drop 以内なら採用）
        accepted = drop <= cfg.select_max_score_drop

        rounds.append(
            {
                "feature": target["name"],
                "importance": importance,
                "val_score": cand_result["val_score"],
                "score_drop": drop,
                "accepted": accepted,
            }
        )
        logger.info(
            "選択ラウンド %d: '%s'（重要度 %.4f）除去 → val %.4f（低下 %+.4f）→ %s",
            len(rounds),
            target["name"],
            importance,
            cand_result["val_score"],
            drop,
            "採用" if accepted else "棄却",
        )

        if not accepted:
            break
        for f in candidate["features"]:
            if f["name"] == target["name"]:
                f["rationale"] = (
                    f"特徴量選択により削除（重要度 {importance:.4f}, val低下 {drop:+.4f}）"
                )
        removed.append(target["name"])
        current_schema = candidate
        current_result = cand_result
    else:
        logger.info("最小特徴量数 %d に到達したため終了", cfg.select_min_features)

    n_after = len(active_features(current_schema))
    logger.info(
        "特徴量選択完了: %d → %d特徴量, val %.4f → %.4f",
        n_before,
        n_after,
        baseline,
        current_result["val_score"],
    )

    selection = {
        "baseline_val_score": baseline,
        "final_val_score": current_result["val_score"],
        "max_score_drop": cfg.select_max_score_drop,
        "min_features": cfg.select_min_features,
        "n_features_before": n_before,
        "n_features_after": n_after,
        "removed": removed,
        "rounds": rounds,
        "schema": current_schema,
        "result": current_result,
    }

    if schema_path is not None:
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(
            json.dumps(current_schema, ensure_ascii=False, indent=2)
        )
    if model_path is not None and removed:
        # 除去ゼロなら合格モデル（model_v{N}.txt）がそのまま最終モデル
        model_path.parent.mkdir(parents=True, exist_ok=True)
        current_result["_model"].booster_.save_model(str(model_path))
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            k: v for k, v in selection.items() if k not in ("schema", "result")
        }
        out_json.write_text(json.dumps(serializable, ensure_ascii=False, indent=2))

    return selection
