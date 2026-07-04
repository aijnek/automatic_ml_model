from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import report


@pytest.fixture
def clf_result() -> dict:
    return {
        "metric_name": "macro_f1",
        "val_score": 0.5,
        "n_train": 10,
        "n_val": 5,
        "val_metrics": {
            "per_class_f1": {"red": 0.5, "green": 0.5},
            "confusion_matrix": {"labels": ["red", "green"], "matrix": [[3, 1], [1, 0]]},
        },
        "feature_importances": {"dominant_color": 0.9, "brightness": 0.1},
        "worst_val_samples": [],
    }


def _features():
    return [
        {"name": "dominant_color", "type": "categorical",
         "choices": ["red", "green"], "prompt": "?", "action": "new", "rationale": ""},
        {"name": "brightness", "type": "scale_1_5", "prompt": "?",
         "action": "new", "rationale": ""},
    ]


def test_compute_distribution_stats():
    df = pd.DataFrame(
        {
            "filename": ["a", "b", "c", "d"],
            "dominant_color": ["red", "red", "red", np.nan],
            "brightness": [3, 3, 3, 3],
        }
    )
    stats = report.compute_distribution_stats(df, _features())
    assert stats["dominant_color"]["nan_rate"] == 0.25
    assert stats["dominant_color"]["mode_fraction"] == 1.0  # 非NaN中は全てred
    assert stats["brightness"]["mode_fraction"] == 1.0
    assert stats["brightness"]["n_unique"] == 1


def test_build_quality_records_statuses(clf_config, clf_result):
    extraction_quality = {
        "features": {
            "dominant_color": {"consistency": 0.9, "cross_agreement": 0.8,
                               "carried_forward": False},
            "brightness": {"consistency": 0.4, "cross_agreement": 0.9,
                           "carried_forward": False},
        }
    }
    dist_stats = {
        "dominant_color": {"nan_rate": 0.0, "mode_fraction": 0.5, "n_unique": 2},
        "brightness": {"nan_rate": 0.0, "mode_fraction": 0.5, "n_unique": 4},
    }
    records = report.build_quality_records(
        clf_config, extraction_quality, dist_stats, _features()
    )
    assert records["dominant_color"]["status"] == "安定"
    assert records["brightness"]["status"] == "不安定"
    assert any("自己一致率" in r for r in records["brightness"]["reasons"])


def test_report_md_includes_quality_section(clf_config, clf_result):
    records = {
        "dominant_color": {"status": "安定", "reasons": [], "consistency": 0.9,
                           "cross_agreement": None, "carried_forward": False,
                           "nan_rate": 0.0, "mode_fraction": 0.5, "n_unique": 2},
    }
    md = report.build_report_md(clf_config, 1, clf_result, records, "ref-model")
    assert "## 抽出信頼性" in md
    assert "ref-model" in md
    assert "| dominant_color | 安定 | 0.90 | - | 0.00 | 0.50 |" in md
    # 信頼性セクションなしの場合は含まれない
    md_plain = report.build_report_md(clf_config, 1, clf_result)
    assert "抽出信頼性" not in md_plain


def test_save_iteration_report_with_quality(tmp_path, clf_config, clf_result):
    df = pd.DataFrame(
        {
            "filename": ["a", "b"],
            "dominant_color": ["red", "green"],
            "brightness": [1, 5],
        }
    )
    schema = {"version": 1, "features": _features()}
    extraction_quality = {
        "reference_model": "ref-model",
        "features": {
            "dominant_color": {"consistency": 1.0, "cross_agreement": 1.0,
                               "carried_forward": False},
            "brightness": {"consistency": 1.0, "cross_agreement": 1.0,
                           "carried_forward": False},
        },
    }
    md = report.save_iteration_report(
        clf_config, 1, clf_result, tmp_path,
        extraction_quality=extraction_quality, features_df=df, schema=schema,
    )
    assert "## 抽出信頼性" in md
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["extraction_quality"]["dominant_color"]["status"] == "安定"


def test_save_iteration_report_without_quality(tmp_path, clf_config, clf_result):
    md = report.save_iteration_report(clf_config, 1, clf_result, tmp_path)
    assert "抽出信頼性" not in md
    saved = json.loads((tmp_path / "report.json").read_text())
    assert "extraction_quality" not in saved
