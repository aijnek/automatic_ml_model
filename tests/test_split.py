from __future__ import annotations

import pandas as pd
import pytest

from pipeline import split as split_mod


def _write_annotations(tmp_path, df):
    path = tmp_path / "annotations.csv"
    df.to_csv(path, index=False)
    return path


def test_split_ratios(tmp_path, clf_config, clf_annotations):
    ann = _write_annotations(tmp_path, clf_annotations)
    splits = split_mod.run(ann, tmp_path / "splits", clf_config)
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15
    # 全サンプルが重複なくどこかに入る
    all_files = pd.concat([s["filename"] for s in splits.values()])
    assert sorted(all_files) == sorted(clf_annotations["filename"])
    assert all_files.is_unique


def test_classification_stratify(tmp_path, clf_config, clf_annotations):
    ann = _write_annotations(tmp_path, clf_annotations)
    splits = split_mod.run(ann, tmp_path / "splits", clf_config)
    overall = clf_annotations["label"].value_counts(normalize=True)
    for name, df in splits.items():
        dist = df["label"].value_counts(normalize=True)
        for cls in overall.index:
            assert abs(dist.get(cls, 0) - overall[cls]) < 0.1, f"{name}/{cls}"


def test_regression_quantile_stratify(tmp_path, reg_config, reg_annotations):
    ann = _write_annotations(tmp_path, reg_annotations)
    splits = split_mod.run(ann, tmp_path / "splits", reg_config)
    overall_median = reg_annotations["label"].median()
    for name, df in splits.items():
        # 各splitの中央値が全体から大きく外れない（分位ビンstratifyの効果）
        assert abs(df["label"].median() - overall_median) < 2.0, name


def test_seed_reproducibility(tmp_path, clf_config, clf_annotations):
    ann = _write_annotations(tmp_path, clf_annotations)
    s1 = split_mod.run(ann, tmp_path / "splits1", clf_config)
    s2 = split_mod.run(ann, tmp_path / "splits2", clf_config)
    for name in split_mod.SPLIT_NAMES:
        pd.testing.assert_frame_equal(s1[name], s2[name])


def test_no_overwrite_without_force(tmp_path, clf_config, clf_annotations):
    ann = _write_annotations(tmp_path, clf_annotations)
    splits_dir = tmp_path / "splits"
    split_mod.run(ann, splits_dir, clf_config)
    with pytest.raises(FileExistsError):
        split_mod.run(ann, splits_dir, clf_config)
    # force なら通る
    split_mod.run(ann, splits_dir, clf_config, force=True)


def test_too_few_annotations(tmp_path, clf_config, clf_annotations):
    ann = _write_annotations(tmp_path, clf_annotations.head(5))
    with pytest.raises(ValueError):
        split_mod.run(ann, tmp_path / "splits", clf_config)
