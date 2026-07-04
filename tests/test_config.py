from __future__ import annotations

import yaml

from pipeline.config import load_config, save_config


def test_config_roundtrip_with_verification(tmp_path, clf_config):
    clf_config.verification_enabled = False
    clf_config.reference_vlm_model = "some-ref-model"
    clf_config.verify_sample_size = 10
    clf_config.verify_consistency_threshold = 0.8
    path = tmp_path / "config.yaml"
    save_config(clf_config, path)
    loaded = load_config(path)
    assert loaded == clf_config


def test_config_defaults_without_verification_block(tmp_path, clf_config):
    path = tmp_path / "config.yaml"
    save_config(clf_config, path)
    data = yaml.safe_load(path.read_text())
    del data["verification"]  # 既存の config.yaml（verification なし）を再現
    path.write_text(yaml.safe_dump(data, allow_unicode=True))

    loaded = load_config(path)
    assert loaded.verification_enabled is True
    assert loaded.verify_sample_size == 25
    assert loaded.verify_n_repeats == 3
    assert loaded.verify_nan_rate_threshold == 0.2


def test_config_roundtrip_with_feature_selection(tmp_path, clf_config):
    clf_config.feature_selection_enabled = False
    clf_config.select_max_score_drop = 0.05
    clf_config.select_min_features = 2
    path = tmp_path / "config.yaml"
    save_config(clf_config, path)
    loaded = load_config(path)
    assert loaded == clf_config


def test_config_defaults_without_feature_selection_block(tmp_path, clf_config):
    path = tmp_path / "config.yaml"
    save_config(clf_config, path)
    data = yaml.safe_load(path.read_text())
    del data["feature_selection"]  # 既存の config.yaml（セクションなし）を再現
    path.write_text(yaml.safe_dump(data, allow_unicode=True))

    loaded = load_config(path)
    assert loaded.feature_selection_enabled is True
    assert loaded.select_max_score_drop == 0.01
    assert loaded.select_min_features == 1
