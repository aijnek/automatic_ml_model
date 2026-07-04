from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import Config


@pytest.fixture
def clf_config() -> Config:
    return Config(
        task_type="classification",
        description="色付き図形の画像。図形の色を分類する。",
        classes=["red", "green", "blue"],
        threshold=0.9,
        seed=42,
    )


@pytest.fixture
def reg_config() -> Config:
    return Config(
        task_type="regression",
        description="図形が描かれた画像。図形の個数を予測する。",
        target_min=0.0,
        target_max=10.0,
        threshold=0.9,
        seed=42,
    )


@pytest.fixture
def clf_annotations() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    labels = ["red"] * 40 + ["green"] * 30 + ["blue"] * 30
    rng.shuffle(labels)
    return pd.DataFrame(
        {"filename": [f"img_{i:03d}.png" for i in range(100)], "label": labels}
    )


@pytest.fixture
def reg_annotations() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "filename": [f"img_{i:03d}.png" for i in range(100)],
            "label": rng.uniform(0, 10, size=100).round(2),
        }
    )


@pytest.fixture
def sample_schema() -> dict:
    return {
        "version": 1,
        "features": [
            {
                "name": "dominant_color",
                "type": "categorical",
                "choices": ["red", "green", "blue", "other"],
                "prompt": "What is the dominant color of the shape?",
                "action": "new",
                "rationale": "色がタスクの主対象",
            },
            {
                "name": "brightness",
                "type": "scale_1_5",
                "prompt": "Rate the overall brightness from 1 (dark) to 5 (bright).",
                "action": "new",
                "rationale": "照明条件の影響を捉える",
            },
            {
                "name": "has_multiple_shapes",
                "type": "binary",
                "prompt": "Does the image contain more than one shape? true/false.",
                "action": "new",
                "rationale": "複数図形の混在検出",
            },
            {
                "name": "shape_area_ratio",
                "type": "float",
                "prompt": "Estimate the fraction of the image occupied by shapes (0.0-1.0).",
                "action": "new",
                "rationale": "図形サイズの proxy",
            },
        ],
    }
