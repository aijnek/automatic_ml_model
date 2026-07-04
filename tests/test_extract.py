from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from pipeline import extract
from pipeline.designer import active_features


def _fake_vlm(answers: dict[str, object]):
    """要求されたスキーマのキーにだけ answers から値を返すモックVLM。"""
    calls = []

    def fn(model, prompt, image_path, format_schema):
        calls.append({"image": image_path.name, "keys": list(format_schema["properties"])})
        return json.dumps(
            {k: answers[k] for k in format_schema["properties"] if k in answers}
        )

    fn.calls = calls
    return fn


ANSWERS = {
    "dominant_color": "red",
    "brightness": 4,
    "has_multiple_shapes": False,
    "shape_area_ratio": 0.35,
}


# ---- schema_to_json_schema ----

def test_json_schema_generation(sample_schema):
    js = extract.schema_to_json_schema(active_features(sample_schema))
    props = js["properties"]
    assert props["brightness"] == {"type": "integer", "minimum": 1, "maximum": 5}
    assert props["has_multiple_shapes"] == {"type": "boolean"}
    assert props["dominant_color"]["enum"] == ["red", "green", "blue", "other"]
    assert props["shape_area_ratio"] == {"type": "number"}
    assert set(js["required"]) == set(props)


# ---- coerce_value ----

@pytest.mark.parametrize(
    "type_,value,expected",
    [
        ("binary", True, 1),
        ("binary", "false", 0),
        ("scale_1_5", 3, 3),
        ("scale_1_5", "4", 4),
        ("scale_1_5", 9, np.nan),
        ("float", "0.5", 0.5),
        ("float", "high", np.nan),
        ("categorical", "red", "red"),
        ("categorical", "purple", np.nan),
        ("categorical", None, np.nan),
    ],
)
def test_coerce_value(type_, value, expected):
    feature = {"name": "x", "type": type_, "choices": ["red", "green"]}
    result = extract.coerce_value(feature, value)
    if isinstance(expected, float) and np.isnan(expected):
        assert result is np.nan or (isinstance(result, float) and np.isnan(result))
    else:
        assert result == expected


# ---- features_to_extract（差分判定） ----

def test_features_to_extract_initial(sample_schema):
    assert len(extract.features_to_extract(sample_schema, None)) == 4


def test_features_to_extract_diff(sample_schema):
    new_schema = copy.deepcopy(sample_schema)
    new_schema["version"] = 2
    for f in new_schema["features"]:
        f["action"] = "kept"
    # brightness のプロンプトだけ変更
    next(f for f in new_schema["features"] if f["name"] == "brightness")[
        "prompt"
    ] = "Rate brightness 1 (very dark) to 5 (overexposed)."
    changed = extract.features_to_extract(new_schema, sample_schema)
    assert [f["name"] for f in changed] == ["brightness"]


# ---- run_extraction ----

@pytest.fixture
def images(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    names = [f"img_{i}.png" for i in range(5)]
    for n in names:
        (images_dir / n).write_bytes(b"fake")
    return images_dir, names


def test_run_extraction_basic(tmp_path, sample_schema, images):
    images_dir, names = images
    vlm = _fake_vlm(ANSWERS)
    df = extract.run_extraction(
        sample_schema, names, images_dir, tmp_path / "f_v1.csv", "test-model", vlm_fn=vlm
    )
    assert len(df) == 5
    assert df["dominant_color"].eq("red").all()
    assert df["brightness"].eq(4).all()
    assert df["has_multiple_shapes"].eq(0).all()
    assert len(vlm.calls) == 5


def test_run_extraction_reuses_cached_columns(tmp_path, sample_schema, images):
    images_dir, names = images
    v1_csv = tmp_path / "f_v1.csv"
    extract.run_extraction(
        sample_schema, names, images_dir, v1_csv, "m", vlm_fn=_fake_vlm(ANSWERS)
    )

    schema_v2 = copy.deepcopy(sample_schema)
    schema_v2["version"] = 2
    next(f for f in schema_v2["features"] if f["name"] == "brightness")[
        "prompt"
    ] = "changed prompt"

    vlm2 = _fake_vlm({"brightness": 2})
    df = extract.run_extraction(
        schema_v2,
        names,
        images_dir,
        tmp_path / "f_v2.csv",
        "m",
        prev_schema=sample_schema,
        prev_csv=v1_csv,
        vlm_fn=vlm2,
    )
    # 変更された brightness のみVLMに問い合わせる
    assert all(c["keys"] == ["brightness"] for c in vlm2.calls)
    assert df["brightness"].eq(2).all()
    # 変更なしの列はキャッシュから流用される
    assert df["dominant_color"].eq("red").all()
    assert df["shape_area_ratio"].eq(0.35).all()


def test_run_extraction_resume_skips_done(tmp_path, sample_schema, images):
    images_dir, names = images
    out_csv = tmp_path / "f_v1.csv"
    extract.run_extraction(
        sample_schema, names[:2], images_dir, out_csv, "m", vlm_fn=_fake_vlm(ANSWERS)
    )
    vlm2 = _fake_vlm(ANSWERS)
    df = extract.run_extraction(
        sample_schema, names, images_dir, out_csv, "m", vlm_fn=vlm2
    )
    # 処理済み2枚はスキップされ、残り3枚だけVLMが呼ばれる
    assert len(vlm2.calls) == 3
    assert len(df) == 5


def test_run_extraction_failure_fills_nan(tmp_path, sample_schema, images):
    images_dir, names = images

    def broken_vlm(model, prompt, image_path, format_schema):
        raise ConnectionError("ollama down")

    df = extract.run_extraction(
        sample_schema, names, images_dir, tmp_path / "f.csv", "m", vlm_fn=broken_vlm
    )
    assert len(df) == 5
    assert df["brightness"].isna().all()


def test_parse_vlm_json_repairs_unquoted_yes():
    raw = '{\n  "dominant_color": "red",\n  "has_multiple_shapes": yes\n}'
    parsed = extract.parse_vlm_json(raw)
    assert parsed["has_multiple_shapes"] == "yes"


def test_parse_vlm_json_extracts_from_prose():
    raw = 'Sure! Here is the answer:\n{"brightness": 4}\nLet me know.'
    assert extract.parse_vlm_json(raw)["brightness"] == 4


def test_extract_one_image_fallback_per_feature(tmp_path, sample_schema):
    """バッチが空応答でも、単項目フォールバックで値が埋まる。"""
    img = tmp_path / "img.png"
    img.write_bytes(b"fake")
    feats = [f for f in sample_schema["features"]]
    calls = []

    def vlm(model, prompt, image_path, format_schema):
        keys = list(format_schema["properties"])
        calls.append(keys)
        if len(keys) > 1:  # バッチは空応答（実機で観測された故障モード）
            return ""
        return json.dumps({keys[0]: ANSWERS[keys[0]]})

    row = extract.extract_one_image(feats, img, "m", vlm_fn=vlm)
    assert row["dominant_color"] == "red"
    assert row["brightness"] == 4
    # バッチ2回（リトライ含む）+ 単項目4回
    assert sum(1 for k in calls if len(k) > 1) == 2
    assert sum(1 for k in calls if len(k) == 1) == 4


def test_run_extraction_retries_all_nan_rows_on_resume(tmp_path, sample_schema, images):
    images_dir, names = images
    out_csv = tmp_path / "f.csv"

    def broken_vlm(model, prompt, image_path, format_schema):
        raise ConnectionError("ollama down")

    extract.run_extraction(
        sample_schema, names, images_dir, out_csv, "m", vlm_fn=broken_vlm
    )
    # 全滅（全行NaN）→ 復旧後の再実行で全画像が再抽出され、最新行が採用される
    vlm = _fake_vlm(ANSWERS)
    df = extract.run_extraction(
        sample_schema, names, images_dir, out_csv, "m", vlm_fn=vlm
    )
    assert len(vlm.calls) == 5
    assert len(df) == 5
    assert df["dominant_color"].eq("red").all()


def test_run_extraction_removed_feature_absent(tmp_path, sample_schema, images):
    images_dir, names = images
    schema_v2 = copy.deepcopy(sample_schema)
    next(f for f in schema_v2["features"] if f["name"] == "brightness")[
        "action"
    ] = "removed"
    df = extract.run_extraction(
        schema_v2, names, images_dir, tmp_path / "f.csv", "m", vlm_fn=_fake_vlm(ANSWERS)
    )
    assert "brightness" not in df.columns
