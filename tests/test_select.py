from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pipeline import select as select_mod
from pipeline import train as train_mod
from pipeline.designer import active_features


def _make_features(clf_annotations, informative=True):
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


def _splits(annotations, with_test=True):
    splits = {
        "train": annotations.iloc[:70].reset_index(drop=True),
        "val": annotations.iloc[70:85].reset_index(drop=True),
    }
    if with_test:
        splits["test"] = annotations.iloc[85:].reset_index(drop=True)
    return splits


@pytest.fixture
def clf_setup(clf_config, clf_annotations, sample_schema):
    features = _make_features(clf_annotations)
    splits = _splits(clf_annotations)
    baseline = train_mod.train_and_evaluate(clf_config, sample_schema, features, splits)
    return clf_config, sample_schema, features, splits, baseline


def test_removes_uninformative_features(clf_setup, tmp_path):
    cfg, schema, features, splits, baseline = clf_setup
    out_json = tmp_path / "feature_selection.json"
    schema_path = tmp_path / "schema_selected.json"
    model_path = tmp_path / "model_selected.txt"
    selection = select_mod.select_features(
        cfg, schema, features, splits, baseline,
        out_json=out_json, schema_path=schema_path, model_path=model_path,
    )

    active = [f["name"] for f in active_features(selection["schema"])]
    assert "dominant_color" in active  # 支配的な特徴量は残る
    assert selection["removed"], "ノイズ特徴量が少なくとも1つ除去されるはず"
    assert selection["final_val_score"] >= baseline["val_score"] - cfg.select_max_score_drop
    for f in selection["schema"]["features"]:
        if f["name"] in selection["removed"]:
            assert f["action"] == "removed"
            assert "特徴量選択" in f["rationale"]
    assert selection["rounds"]
    assert out_json.exists()
    assert schema_path.exists()
    assert model_path.exists()  # 除去ありならモデルも保存される


def test_stops_on_first_rejection(clf_config, clf_annotations, sample_schema):
    """ノイズ除去は採用され、必要な特徴量の除去は棄却されて停止する。

    ラベルを shape_area_ratio と has_multiple_shapes の2特徴量で決まるように
    合成する。どちらか一方を除去すると精度が大きく落ちるため必ず棄却される。
    """
    rng = np.random.default_rng(3)
    n = len(clf_annotations)
    area = rng.uniform(0, 1, size=n)
    multi = rng.integers(0, 2, size=n)
    labels = np.where(area > 0.5, np.where(multi == 1, "red", "green"), "blue")
    annotations = clf_annotations.assign(label=labels)
    features = pd.DataFrame(
        {
            "filename": annotations["filename"],
            "dominant_color": rng.choice(["red", "green", "blue", "other"], size=n),
            "brightness": rng.integers(1, 6, size=n),
            "has_multiple_shapes": multi,
            "shape_area_ratio": area,
        }
    )
    splits = _splits(annotations)
    baseline = train_mod.train_and_evaluate(
        clf_config, sample_schema, features, splits
    )

    selection = select_mod.select_features(
        clf_config, sample_schema, features, splits, baseline
    )
    assert selection["rounds"][-1]["accepted"] is False  # 棄却で停止
    active = [f["name"] for f in active_features(selection["schema"])]
    assert {"shape_area_ratio", "has_multiple_shapes"} <= set(active)
    assert set(selection["removed"]) <= {"dominant_color", "brightness"}
    # 棄却された候補は必要な2特徴量のどちらかのはず
    assert selection["rounds"][-1]["feature"] in (
        "shape_area_ratio",
        "has_multiple_shapes",
    )


def test_min_features_respected(clf_setup):
    cfg, schema, features, splits, baseline = clf_setup
    cfg.select_min_features = 3
    selection = select_mod.select_features(cfg, schema, features, splits, baseline)
    assert selection["n_features_after"] >= 3
    assert len(selection["removed"]) <= 1


def test_single_active_feature_noop(clf_config, clf_annotations, sample_schema):
    schema = deepcopy(sample_schema)
    for f in schema["features"]:
        if f["name"] != "dominant_color":
            f["action"] = "removed"
    features = _make_features(clf_annotations)
    splits = _splits(clf_annotations)
    baseline = train_mod.train_and_evaluate(clf_config, schema, features, splits)

    selection = select_mod.select_features(
        clf_config, schema, features, splits, baseline
    )
    assert selection["rounds"] == []
    assert selection["removed"] == []
    assert selection["result"] is baseline
    assert selection["n_features_before"] == selection["n_features_after"] == 1


def test_never_touches_test_split(clf_config, clf_annotations, sample_schema):
    """test キーなしの splits で動くこと = 選択中に test を参照しない証明。"""
    features = _make_features(clf_annotations)
    splits = _splits(clf_annotations, with_test=False)
    baseline = train_mod.train_and_evaluate(
        clf_config, sample_schema, features, splits
    )
    selection = select_mod.select_features(
        clf_config, sample_schema, features, splits, baseline
    )
    assert selection["n_features_after"] <= selection["n_features_before"]


def test_regression_selection(reg_config, reg_annotations, sample_schema):
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
    splits = _splits(reg_annotations)
    baseline = train_mod.train_and_evaluate(
        reg_config, sample_schema, features, splits
    )
    selection = select_mod.select_features(
        reg_config, sample_schema, features, splits, baseline
    )
    active = [f["name"] for f in active_features(selection["schema"])]
    assert "shape_area_ratio" in active  # 情報のある特徴量は残る
    assert selection["removed"], "ノイズ特徴量は除去されるはず"


def test_output_json_serializable(clf_setup, tmp_path):
    cfg, schema, features, splits, baseline = clf_setup
    out_json = tmp_path / "feature_selection.json"
    select_mod.select_features(
        cfg, schema, features, splits, baseline, out_json=out_json
    )
    saved = json.loads(out_json.read_text())
    assert {
        "baseline_val_score",
        "final_val_score",
        "rounds",
        "removed",
        "n_features_before",
        "n_features_after",
    } <= set(saved)
    assert "schema" not in saved
    assert "result" not in saved


def test_all_removals_rejected(clf_setup, tmp_path):
    """許容低下を負にすれば最初のラウンドで必ず棄却 → baseline パススルー。"""
    cfg, schema, features, splits, baseline = clf_setup
    cfg.select_max_score_drop = -1.0  # どんな候補も棄却される
    model_path = tmp_path / "model_selected.txt"
    selection = select_mod.select_features(
        cfg, schema, features, splits, baseline, model_path=model_path
    )
    assert selection["removed"] == []
    assert len(selection["rounds"]) == 1
    assert selection["rounds"][0]["accepted"] is False
    assert selection["result"] is baseline
    assert selection["final_val_score"] == baseline["val_score"]
    assert not model_path.exists()  # 除去ゼロならモデルは保存しない
