"""プロジェクト設定とパス規約。

config.yaml は Streamlit アプリ（問題設定ページ）が生成し、
パイプライン各ステップはここ経由で読み込む。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
SPLITS_DIR = DATA_DIR / "splits"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
STATE_PATH = PROJECT_ROOT / "state.json"

DEFAULT_DESIGNER_COMMAND = [
    "ollama",
    "launch",
    "claude",
    "--model",
    "qwen3.6:35b-a3b-coding-nvfp4",
    "--",
    "-p",
]
DEFAULT_VLM_MODEL = "qwen3.5:9b"


@dataclass
class Config:
    task_type: str  # "classification" | "regression"
    description: str
    classes: list[str] = field(default_factory=list)  # classification のみ
    target_min: float | None = None  # regression のみ
    target_max: float | None = None
    target_unit: str = ""
    threshold: float = 0.9  # val macro-F1 / R² の合格ライン
    seed: int = 42
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    designer_command: list[str] = field(
        default_factory=lambda: list(DEFAULT_DESIGNER_COMMAND)
    )
    vlm_model: str = DEFAULT_VLM_MODEL

    @property
    def is_classification(self) -> bool:
        return self.task_type == "classification"

    @property
    def metric_name(self) -> str:
        return "macro_f1" if self.is_classification else "r2"

    def validate(self) -> None:
        if self.task_type not in ("classification", "regression"):
            raise ValueError(f"unknown task_type: {self.task_type}")
        if self.is_classification and len(self.classes) < 2:
            raise ValueError("classification には classes が2つ以上必要です")


def save_config(cfg: Config, path: Path = CONFIG_PATH) -> None:
    cfg.validate()
    data = {
        "task": {
            "type": cfg.task_type,
            "description": cfg.description,
            "classes": cfg.classes,
            "target_range": {
                "min": cfg.target_min,
                "max": cfg.target_max,
                "unit": cfg.target_unit,
            },
        },
        "training": {
            "target_metric_threshold": cfg.threshold,
            "seed": cfg.seed,
            "split": {
                "train": cfg.split_ratios[0],
                "val": cfg.split_ratios[1],
                "test": cfg.split_ratios[2],
            },
        },
        "llm": {
            "designer_command": cfg.designer_command,
            "vlm_model": cfg.vlm_model,
        },
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def load_config(path: Path = CONFIG_PATH) -> Config:
    data = yaml.safe_load(path.read_text())
    task = data["task"]
    training = data.get("training", {})
    llm = data.get("llm", {})
    split = training.get("split", {})
    target = task.get("target_range") or {}
    cfg = Config(
        task_type=task["type"],
        description=task.get("description", ""),
        classes=list(task.get("classes") or []),
        target_min=target.get("min"),
        target_max=target.get("max"),
        target_unit=target.get("unit") or "",
        threshold=float(training.get("target_metric_threshold", 0.9)),
        seed=int(training.get("seed", 42)),
        split_ratios=(
            float(split.get("train", 0.7)),
            float(split.get("val", 0.15)),
            float(split.get("test", 0.15)),
        ),
        designer_command=list(llm.get("designer_command") or DEFAULT_DESIGNER_COMMAND),
        vlm_model=llm.get("vlm_model", DEFAULT_VLM_MODEL),
    )
    cfg.validate()
    return cfg
