"""LightGBM によるメタデータ→分類/回帰の学習と評価（step 6）。"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)

from pipeline.config import Config
from pipeline.designer import active_features

logger = logging.getLogger(__name__)

# 少数データ前提の控えめな固定ハイパーパラメータ。
# min_data_per_group / cat_smooth はデフォルトだと ~100行規模のデータで
# カテゴリ特徴量の分割が一切作られなくなるため必須。
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=5,
    min_data_in_bin=1,
    min_data_per_group=1,
    cat_smooth=1.0,
    cat_l2=1.0,
    max_cat_to_onehot=8,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 50
N_WORST_SAMPLES = 10


def prepare_matrix(
    features_df: pd.DataFrame, split_df: pd.DataFrame, schema: dict
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """特徴量CSVと split(filename,label) を結合し X, y, filenames を返す。"""
    feats = active_features(schema)
    merged = split_df.merge(features_df, on="filename", how="left", validate="1:1")
    X = merged[[f["name"] for f in feats]].copy()
    for f in feats:
        if f["type"] == "categorical":
            X[f["name"]] = pd.Categorical(X[f["name"]], categories=f["choices"])
        else:
            X[f["name"]] = pd.to_numeric(X[f["name"]], errors="coerce")
    return X, merged["label"], merged["filename"]


def _evaluate(cfg: Config, y_true, y_pred) -> dict:
    if cfg.is_classification:
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "per_class_f1": {
                cls: float(f)
                for cls, f in zip(
                    cfg.classes,
                    f1_score(y_true, y_pred, average=None, labels=cfg.classes),
                )
            },
            "confusion_matrix": {
                "labels": cfg.classes,
                "matrix": confusion_matrix(
                    y_true, y_pred, labels=cfg.classes
                ).tolist(),
            },
        }
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "residual_stats": {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "max_abs": float(np.abs(residuals).max()),
        },
    }


def _worst_samples(
    cfg: Config, X_val: pd.DataFrame, y_true, y_pred, filenames
) -> list[dict]:
    """designer へのフィードバック用: 誤分類 / 高誤差サンプルの特徴量値。"""
    records = []
    if cfg.is_classification:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        idx = np.where(y_true != y_pred)[0]
        order = idx  # 分類は順序なし、先頭からN件
    else:
        errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        order = np.argsort(-errors)
    for i in list(order)[:N_WORST_SAMPLES]:
        rec = {
            "filename": str(filenames.iloc[i]),
            "true": _to_py(np.asarray(y_true)[i]),
            "predicted": _to_py(np.asarray(y_pred)[i]),
            "features": {k: _to_py(v) for k, v in X_val.iloc[i].items()},
        }
        records.append(rec)
    return records


def _to_py(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    if pd.isna(v):
        return None
    return v


def train_and_evaluate(
    cfg: Config,
    schema: dict,
    features_df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    model_path: Path | None = None,
) -> dict:
    """train で学習し val で評価。診断情報を含む結果 dict を返す。"""
    X_train, y_train, _ = prepare_matrix(features_df, splits["train"], schema)
    X_val, y_val, val_files = prepare_matrix(features_df, splits["val"], schema)

    if cfg.is_classification:
        model = lgb.LGBMClassifier(random_state=cfg.seed, **LGBM_PARAMS)
    else:
        model = lgb.LGBMRegressor(random_state=cfg.seed, **LGBM_PARAMS)
        y_train = pd.to_numeric(y_train)
        y_val = pd.to_numeric(y_val)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    y_pred = model.predict(X_val)

    metrics = _evaluate(cfg, y_val, y_pred)
    val_score = metrics[cfg.metric_name]

    importances = model.booster_.feature_importance(importance_type="gain")
    total = importances.sum() or 1.0
    importance_map = {
        name: float(imp / total)
        for name, imp in sorted(
            zip(model.booster_.feature_name(), importances),
            key=lambda x: -x[1],
        )
    }

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(model_path))

    result = {
        "task_type": cfg.task_type,
        "metric_name": cfg.metric_name,
        "val_score": val_score,
        "val_metrics": metrics,
        "feature_importances": importance_map,
        "worst_val_samples": _worst_samples(cfg, X_val, y_val, y_pred, val_files),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_iteration": int(model.booster_.best_iteration or 0),
        "_model": model,  # test評価用（JSON保存時には除外する）
    }
    logger.info("学習完了: val %s = %.4f", cfg.metric_name, val_score)
    return result


def evaluate_on_test(
    cfg: Config,
    schema: dict,
    features_df: pd.DataFrame,
    test_split: pd.DataFrame,
    model: lgb.LGBMModel,
) -> dict:
    """最終評価。イテレーション中は絶対に呼ばないこと。"""
    X_test, y_test, _ = prepare_matrix(features_df, test_split, schema)
    if not cfg.is_classification:
        y_test = pd.to_numeric(y_test)
    y_pred = model.predict(X_test)
    metrics = _evaluate(cfg, y_test, y_pred)
    return {
        "test_score": metrics[cfg.metric_name],
        "test_metrics": metrics,
        "n_test": len(X_test),
    }
