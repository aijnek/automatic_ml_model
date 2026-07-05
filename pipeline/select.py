"""合格モデルに対する特徴量選択（重要度ベースの後方消去, step 7.5）。

val 合格後・test 評価前に実行する。gain 重要度が最小の特徴量を1つずつ除去して
再学習し、baseline からの低下が cfg.select_max_score_drop 以内なら採用、
超えたら棄却して終了する。

採否判定は既定では train/val の単一分割スコアで行う。cfg.select_cv_enabled を
有効にすると、train+val を合わせた cfg.select_cv_folds 分割の交差検証平均スコアで
判定する（単一の val 分割だけで判定すると、少数データでは val の偶然の当たり外れに
特徴量部分集合がフィットしてしまう＝val への過学習が起きうるため）。
重要度ランキングと最終モデル・報告用スコアは常に train/val の単一分割で得る。

- test split は絶対に参照しない（最終評価は run_loop の step 8 の1回のみ）
- LLM 呼び出しなし・seed 固定で決定的なので、中断時は再実行でやり直すだけでよい
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.designer import active_features
from pipeline.train import make_cv_folds, train_and_evaluate

logger = logging.getLogger(__name__)


def _cv_score(
    cfg: Config,
    schema: dict,
    features_df: pd.DataFrame,
    train_val_df: pd.DataFrame,
) -> float:
    """train+val を合わせた集合を分割し直し、各分割の val スコアの平均を返す。"""
    splitter, split_args = make_cv_folds(cfg, train_val_df, cfg.select_cv_folds)
    scores = []
    for train_idx, val_idx in splitter.split(*split_args):
        fold_splits = {
            "train": train_val_df.iloc[train_idx],
            "val": train_val_df.iloc[val_idx],
        }
        result = train_and_evaluate(cfg, schema, features_df, fold_splits)
        scores.append(result["val_score"])
    return float(np.mean(scores))


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
    use_cv = cfg.select_cv_enabled
    train_val_df = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    current_schema = deepcopy(schema)
    current_result = baseline_result  # 重要度ランキング・最終モデル用（単一分割）
    baseline_score = baseline_result["val_score"]
    baseline_cv = _cv_score(cfg, current_schema, features_df, train_val_df) if use_cv else None
    current_score = baseline_cv if use_cv else baseline_score
    n_before = len(active_features(current_schema))
    rounds: list[dict] = []
    removed: list[str] = []

    logger.info(
        "特徴量選択開始: %d特徴量, baseline %s%s = %.4f（許容低下 %.3f, 最小特徴量数 %d）",
        n_before,
        f"CV({cfg.select_cv_folds}分割) " if use_cv else "val ",
        cfg.metric_name,
        current_score,
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

        if use_cv:
            cand_score = _cv_score(cfg, candidate, features_df, train_val_df)
            cand_result = None
        else:
            cand_result = train_and_evaluate(cfg, candidate, features_df, splits)
            cand_score = cand_result["val_score"]

        drop = current_score - cand_score
        # 許容低下のみで判定する（cfg.threshold を下回っても baseline との差が
        # select_max_score_drop 以内なら採用）
        accepted = drop <= cfg.select_max_score_drop

        rounds.append(
            {
                "feature": target["name"],
                "importance": importance,
                "score": cand_score,
                "score_drop": drop,
                "accepted": accepted,
            }
        )
        logger.info(
            "選択ラウンド %d: '%s'（重要度 %.4f）除去 → %s %.4f（低下 %+.4f）→ %s",
            len(rounds),
            target["name"],
            importance,
            "CV" if use_cv else "val",
            cand_score,
            drop,
            "採用" if accepted else "棄却",
        )

        if not accepted:
            break
        for f in candidate["features"]:
            if f["name"] == target["name"]:
                f["rationale"] = (
                    f"特徴量選択により削除（重要度 {importance:.4f}, スコア低下 {drop:+.4f}）"
                )
        removed.append(target["name"])
        current_schema = candidate
        current_score = cand_score
        # 次ラウンドの重要度ランキング・最終モデル候補は常に単一分割で得る
        current_result = cand_result if cand_result is not None else train_and_evaluate(
            cfg, candidate, features_df, splits
        )
    else:
        logger.info("最小特徴量数 %d に到達したため終了", cfg.select_min_features)

    n_after = len(active_features(current_schema))
    logger.info(
        "特徴量選択完了: %d → %d特徴量, %s %.4f → %.4f（val単一分割: %.4f → %.4f）",
        n_before,
        n_after,
        "CV" if use_cv else "val",
        baseline_cv if use_cv else baseline_score,
        current_score,
        baseline_result["val_score"],
        current_result["val_score"],
    )

    selection = {
        "baseline_val_score": baseline_result["val_score"],
        "final_val_score": current_result["val_score"],
        "cv_enabled": use_cv,
        "baseline_cv_score": baseline_cv,
        "final_cv_score": current_score if use_cv else None,
        "cv_folds": cfg.select_cv_folds if use_cv else None,
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
