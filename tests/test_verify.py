from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pytest

from pipeline import verify

REF_MODEL = "ref-vlm"


@pytest.fixture
def vcfg(clf_config):
    clf_config.reference_vlm_model = REF_MODEL
    clf_config.verify_sample_size = 4
    clf_config.verify_n_repeats = 2
    return clf_config


def _feature(name, type_="scale_1_5", **kw):
    f = {
        "name": name,
        "type": type_,
        "prompt": f"Question about {name}?",
        "action": kw.pop("action", "new"),
        "rationale": "test",
    }
    if type_ == "categorical":
        f.setdefault("choices", kw.pop("choices", ["red", "green", "blue"]))
    f.update(kw)
    return f


# ---- values_agree ----

@pytest.mark.parametrize(
    "type_,a,b,expected",
    [
        ("binary", 1, 1, True),
        ("binary", 1, 0, False),
        ("categorical", "red", "red", True),
        ("categorical", "red", "blue", False),
        ("scale_1_5", 3, 4, True),  # ±1以内は一致
        ("scale_1_5", 3, 5, False),
        ("float", 1.0, 1.1, True),  # 相対誤差15%以内
        ("float", 1.0, 2.0, False),
        ("float", 0.0, 0.0, True),
    ],
)
def test_values_agree(type_, a, b, expected):
    assert verify.values_agree(_feature("f", type_), a, b) is expected


def test_values_agree_nan_cases():
    f = _feature("f", "binary")
    assert verify.values_agree(f, np.nan, np.nan) is None  # 両方NaN → ペア除外
    assert verify.values_agree(f, 1, np.nan) is False  # 片方NaN → 不一致
    assert verify.values_agree(f, None, 1) is False


# ---- consensus_value ----

def test_consensus_value_mode_and_median():
    assert verify.consensus_value(_feature("f", "categorical"), ["red", "red", "blue"]) == "red"
    assert verify.consensus_value(_feature("f", "float"), [1.0, 2.0, 10.0]) == 2.0
    assert verify.consensus_value(_feature("f", "binary"), [1, np.nan, 1]) == 1
    assert np.isnan(verify.consensus_value(_feature("f", "binary"), [np.nan, None]))


# ---- compute_feature_signals ----

def _repeats(values_by_repeat: list[dict[str, object]], name="f"):
    """{filename: {name: value}} × n_repeats を組み立てるヘルパ。"""
    return [
        {fname: {name: v} for fname, v in run.items()} for run in values_by_repeat
    ]


def test_signals_stable_feature():
    f = _feature("f", "categorical")
    repeats = _repeats([{"a": "red", "b": "blue"}, {"a": "red", "b": "blue"}])
    reference = {"a": {"f": "red"}, "b": {"f": "blue"}}
    s = verify.compute_feature_signals(f, repeats, reference)
    assert s["consistency"] == 1.0
    assert s["cross_agreement"] == 1.0
    assert s["sample_nan_rate"] == 0.0
    assert s["n_images"] == 2 and s["n_repeats"] == 2


def test_signals_flapping_feature():
    f = _feature("f", "binary")
    repeats = _repeats([{"a": 1, "b": 1}, {"a": 0, "b": 1}])
    s = verify.compute_feature_signals(f, repeats, None)
    assert s["consistency"] == 0.5  # a: 不一致, b: 一致
    assert s["cross_agreement"] is None  # 参照なし


def test_signals_reference_disagreement():
    f = _feature("f", "categorical")
    repeats = _repeats([{"a": "red"}, {"a": "red"}])
    reference = {"a": {"f": "blue"}}
    s = verify.compute_feature_signals(f, repeats, reference)
    assert s["consistency"] == 1.0
    assert s["cross_agreement"] == 0.0  # ブレないが参照と食い違う


def test_signals_single_repeat_has_no_consistency():
    f = _feature("f", "binary")
    s = verify.compute_feature_signals(f, _repeats([{"a": 1}]), None)
    assert s["consistency"] is None


# ---- classify_status ----

def test_classify_priority_extraction_failure_first(vcfg):
    # NaN率超過は不安定シグナルより優先される
    signals = {"consistency": 0.1, "cross_agreement": 0.1}
    status, reasons = verify.classify_status(
        signals, {"nan_rate": 0.5, "mode_fraction": 0.5}, vcfg
    )
    assert status == "抽出失敗"
    assert "NaN率" in reasons[0]


def test_classify_degenerate(vcfg):
    status, reasons = verify.classify_status(
        {"consistency": 1.0}, {"nan_rate": 0.0, "mode_fraction": 0.99}, vcfg
    )
    assert status == "縮退"


def test_classify_unstable_by_consistency(vcfg):
    status, reasons = verify.classify_status(
        {"consistency": 0.5, "cross_agreement": 0.9},
        {"nan_rate": 0.0, "mode_fraction": 0.4},
        vcfg,
    )
    assert status == "不安定"
    assert any("自己一致率" in r for r in reasons)


def test_classify_unstable_by_cross_agreement(vcfg):
    status, reasons = verify.classify_status(
        {"consistency": 0.9, "cross_agreement": 0.3},
        {"nan_rate": 0.0, "mode_fraction": 0.4},
        vcfg,
    )
    assert status == "不安定"
    assert any("参照VLM一致率" in r for r in reasons)


def test_classify_stable(vcfg):
    status, reasons = verify.classify_status(
        {"consistency": 0.9, "cross_agreement": 0.8},
        {"nan_rate": 0.0, "mode_fraction": 0.4},
        vcfg,
    )
    assert status == "安定" and reasons == []


def test_classify_without_signals_uses_dist_only(vcfg):
    # kept特徴量で過去シグナルが無くても分布統計だけで判定できる
    status, _ = verify.classify_status(None, {"nan_rate": 0.5, "mode_fraction": 0.1}, vcfg)
    assert status == "抽出失敗"
    status, _ = verify.classify_status(None, {"nan_rate": 0.0, "mode_fraction": 0.1}, vcfg)
    assert status == "安定"


# ---- sample_images ----

def test_sample_images_deterministic():
    files = [f"img_{i}.png" for i in range(50)]
    s1 = verify.sample_images(files, 10, seed=42, iteration=1)
    s2 = verify.sample_images(files, 10, seed=42, iteration=1)
    s3 = verify.sample_images(files, 10, seed=42, iteration=2)
    assert s1 == s2
    assert s1 != s3  # イテレーションごとに別サンプル
    assert len(s1) == 10 and len(set(s1)) == 10


def test_sample_images_smaller_population():
    files = ["a.png", "b.png"]
    assert sorted(verify.sample_images(files, 10, 42, 1)) == files


# ---- run_verification ----

class FakeVLM:
    """呼び出しを記録し、stable_feat は常に同値・flappy_feat は呼び出しごとに変わる。"""

    def __init__(self, ref_fails=False):
        self.n_calls = 0
        self.models: list[str] = []
        self.asked_features: set[str] = set()
        self.ref_fails = ref_fails
        self._per_file = defaultdict(int)

    def __call__(self, model, prompt, image_path, format_schema, **_):
        self.n_calls += 1
        self.models.append(model)
        self.asked_features |= set(format_schema["properties"])
        if self.ref_fails and model == REF_MODEL:
            raise RuntimeError("model not found")
        c = self._per_file[image_path.name]
        self._per_file[image_path.name] += 1
        out = {}
        for key in format_schema["properties"]:
            out[key] = 3 if key == "stable_feat" else (1 if c % 2 == 0 else 5)
        return json.dumps(out)


def _make_env(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    filenames = [f"img_{i:02d}.png" for i in range(10)]
    for name in filenames:
        (images_dir / name).write_bytes(b"fake")
    return images_dir, filenames


def test_run_verification_differential_and_carry_forward(tmp_path, vcfg):
    images_dir, filenames = _make_env(tmp_path)
    prev_schema = {"version": 1, "features": [_feature("stable_feat", action="new")]}
    schema = {
        "version": 2,
        "features": [
            _feature("stable_feat", action="kept"),
            _feature("flappy_feat", action="new"),
        ],
    }
    prev_json = tmp_path / "iter1" / "extraction_quality.json"
    prev_json.parent.mkdir()
    prev_json.write_text(json.dumps({
        "features": {"stable_feat": {"consistency": 0.9, "cross_agreement": 0.8,
                                     "carried_forward": False}}
    }))

    fake = FakeVLM()
    result = verify.run_verification(
        schema, prev_schema, filenames, images_dir, vcfg, iteration=2,
        out_json=tmp_path / "iter2" / "extraction_quality.json",
        prev_json=prev_json, vlm_fn=fake,
    )

    # kept はキャリーフォワード、new のみ実測
    assert result["features"]["stable_feat"]["carried_forward"] is True
    assert result["features"]["stable_feat"]["consistency"] == 0.9
    flappy = result["features"]["flappy_feat"]
    assert flappy["carried_forward"] is False
    assert flappy["consistency"] == 0.0  # 呼び出しごとに 1/5 が交互 → 不一致
    # 検証対象は flappy のみ（kept の stable_feat は再抽出されない）
    assert fake.asked_features == {"flappy_feat"}
    assert (tmp_path / "iter2" / "extraction_quality.json").exists()
    assert result["reference_available"] is True
    assert flappy["cross_agreement"] is not None


def test_run_verification_skips_when_json_exists(tmp_path, vcfg):
    images_dir, filenames = _make_env(tmp_path)
    schema = {"version": 1, "features": [_feature("flappy_feat")]}
    out_json = tmp_path / "extraction_quality.json"

    fake = FakeVLM()
    verify.run_verification(
        schema, None, filenames, images_dir, vcfg, 1, out_json, vlm_fn=fake
    )
    n = fake.n_calls
    assert n > 0

    # 2回目は既存JSONを返すだけで VLM を呼ばない（再開）
    result = verify.run_verification(
        schema, None, filenames, images_dir, vcfg, 1, out_json, vlm_fn=fake
    )
    assert fake.n_calls == n
    assert "flappy_feat" in result["features"]


def test_run_verification_reference_failure_is_tolerated(tmp_path, vcfg):
    images_dir, filenames = _make_env(tmp_path)
    schema = {"version": 1, "features": [_feature("stable_feat")]}

    fake = FakeVLM(ref_fails=True)
    result = verify.run_verification(
        schema, None, filenames, images_dir, vcfg, 1,
        out_json=tmp_path / "extraction_quality.json", vlm_fn=fake,
    )
    assert result["reference_available"] is False
    assert result["features"]["stable_feat"]["cross_agreement"] is None
    assert result["features"]["stable_feat"]["consistency"] == 1.0


def test_run_verification_no_targets_is_free(tmp_path, vcfg):
    images_dir, filenames = _make_env(tmp_path)
    schema = {"version": 2, "features": [_feature("stable_feat", action="kept")]}
    prev_schema = {"version": 1, "features": [_feature("stable_feat")]}

    fake = FakeVLM()
    result = verify.run_verification(
        schema, prev_schema, filenames, images_dir, vcfg, 2,
        out_json=tmp_path / "extraction_quality.json", vlm_fn=fake,
    )
    assert fake.n_calls == 0  # スキーマ収束後はコストゼロ
    assert result["sampled"] == []
