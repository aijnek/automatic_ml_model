from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import report as report_mod
from pipeline import train as train_mod


def _make_features(clf_annotations, sample_schema, informative=True):
    """dominant_color がラベルとほぼ一致する合成特徴量を作る。"""
    rng = np.random.default_rng(1)
    n = len(clf_annotations)
    labels = clf_annotations["label"].to_numpy()
    if informative:
        color = labels.copy()
        noise_idx = rng.choice(n, size=max(1, n // 20), replace=False)
        color[noise_idx] = "other"
    else:
        color = rng.choice(["red", "green", "blue", "other"], size=n)
    return pd.DataFrame(
        {
            "filename": clf_annotations["filename"],
            "dominant_color": color,
            "brightness": rng.integers(1, 6, size=n),
            "has_multiple_shapes": rng.integers(0, 2, size=n),
            "shape_area_ratio": rng.uniform(0, 1, size=n),
        }
    )


def _splits(annotations):
    return {
        "train": annotations.iloc[:70].reset_index(drop=True),
        "val": annotations.iloc[70:85].reset_index(drop=True),
        "test": annotations.iloc[85:].reset_index(drop=True),
    }


@pytest.fixture
def clf_setup(clf_config, clf_annotations, sample_schema):
    features = _make_features(clf_annotations, sample_schema)
    return clf_config, sample_schema, features, _splits(clf_annotations)


def test_classification_training(clf_setup):
    cfg, schema, features, splits = clf_setup
    result = train_mod.train_and_evaluate(cfg, schema, features, splits)
    assert result["metric_name"] == "macro_f1"
    assert result["val_score"] > 0.8  # ほぼ決定的な特徴量なので高精度のはず
    assert set(result["val_metrics"]["per_class_f1"]) == set(cfg.classes)
    # 支配的な特徴量の重要度が最大
    top = next(iter(result["feature_importances"]))
    assert top == "dominant_color"


def test_classification_uninformative_features(clf_config, clf_annotations, sample_schema):
    features = _make_features(clf_annotations, sample_schema, informative=False)
    result = train_mod.train_and_evaluate(
        clf_config, sample_schema, features, _splits(clf_annotations)
    )
    assert result["val_score"] < 0.8  # 無情報特徴量では合格ラインに達しない


def test_regression_training(reg_config, reg_annotations, sample_schema):
    rng = np.random.default_rng(2)
    n = len(reg_annotations)
    y = reg_annotations["label"].to_numpy(dtype=float)
    features = pd.DataFrame(
        {
            "filename": reg_annotations["filename"],
            "dominant_color": rng.choice(["red", "green"], size=n),
            "brightness": rng.integers(1, 6, size=n),
            "has_multiple_shapes": rng.integers(0, 2, size=n),
            "shape_area_ratio": y / 10 + rng.normal(0, 0.02, size=n),  # 強い相関
        }
    )
    result = train_mod.train_and_evaluate(
        reg_config, sample_schema, features, _splits(reg_annotations)
    )
    assert result["metric_name"] == "r2"
    assert result["val_score"] > 0.8
    assert "rmse" in result["val_metrics"]


def test_nan_features_tolerated(clf_setup):
    cfg, schema, features, splits = clf_setup
    features = features.copy()
    features.loc[::3, "brightness"] = np.nan
    features.loc[::5, "dominant_color"] = np.nan
    result = train_mod.train_and_evaluate(cfg, schema, features, splits)
    assert result["val_score"] > 0.5


def test_model_saved(clf_setup, tmp_path):
    cfg, schema, features, splits = clf_setup
    model_path = tmp_path / "model.txt"
    train_mod.train_and_evaluate(cfg, schema, features, splits, model_path=model_path)
    assert model_path.exists() and model_path.stat().st_size > 0


def test_evaluate_on_test(clf_setup):
    cfg, schema, features, splits = clf_setup
    result = train_mod.train_and_evaluate(cfg, schema, features, splits)
    test_result = train_mod.evaluate_on_test(
        cfg, schema, features, splits["test"], result["_model"]
    )
    assert 0.0 <= test_result["test_score"] <= 1.0
    assert test_result["n_test"] == len(splits["test"])


def test_report_saved_and_serializable(clf_setup, tmp_path):
    cfg, schema, features, splits = clf_setup
    result = train_mod.train_and_evaluate(cfg, schema, features, splits)
    md = report_mod.save_iteration_report(cfg, 1, result, tmp_path / "iter1")
    saved = json.loads((tmp_path / "iter1" / "report.json").read_text())
    assert "val_score" in saved
    assert "_model" not in saved  # モデルオブジェクトはJSONに含めない
    assert "特徴量重要度" in md
    assert "混同行列" in md
    assert (tmp_path / "iter1" / "report.md").exists()


def test_worst_samples_have_features(clf_config, clf_annotations, sample_schema):
    features = _make_features(clf_annotations, sample_schema, informative=False)
    result = train_mod.train_and_evaluate(
        clf_config, sample_schema, features, _splits(clf_annotations)
    )
    worst = result["worst_val_samples"]
    assert worst, "無情報特徴量なら誤分類が出るはず"
    rec = worst[0]
    assert {"filename", "true", "predicted", "features"} <= set(rec)
    assert "dominant_color" in rec["features"]
    json.dumps(worst)  # JSONシリアライズ可能であること
