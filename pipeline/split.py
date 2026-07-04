"""annotations.csv を train/val/test に分割する（step 3）。

- 分類: ラベルで stratify
- 回帰: ターゲットの分位ビンで擬似 stratify
- 一度作った split は --force なしでは上書きしない（test リーク防止）

実行: uv run python -m pipeline.split [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import ANNOTATIONS_CSV, SPLITS_DIR, Config, load_config

SPLIT_NAMES = ("train", "val", "test")


def _stratify_labels(df: pd.DataFrame, cfg: Config) -> pd.Series | None:
    if cfg.is_classification:
        return df["label"]
    # 回帰: 分位ビン。サンプルが少ない場合はビン数を落とし、それでも無理ならNone
    n_bins = min(5, max(2, len(df) // 20))
    try:
        binned = pd.qcut(df["label"].astype(float), q=n_bins, duplicates="drop")
        if binned.value_counts().min() >= 2:
            return binned
    except ValueError:
        pass
    return None


def make_splits(
    annotations: pd.DataFrame, cfg: Config
) -> dict[str, pd.DataFrame]:
    ratios = cfg.split_ratios
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split比率の合計が1ではありません: {ratios}")

    strat = _stratify_labels(annotations, cfg)
    df_train, df_rest = train_test_split(
        annotations,
        test_size=ratios[1] + ratios[2],
        stratify=strat,
        random_state=cfg.seed,
    )
    strat_rest = None
    if strat is not None:
        strat_rest = strat.loc[df_rest.index]
        # 残り30%側で1件しかないクラスがあるとstratifyできない
        if strat_rest.value_counts().min() < 2:
            strat_rest = None
    df_val, df_test = train_test_split(
        df_rest,
        test_size=ratios[2] / (ratios[1] + ratios[2]),
        stratify=strat_rest,
        random_state=cfg.seed,
    )
    return {
        "train": df_train.reset_index(drop=True),
        "val": df_val.reset_index(drop=True),
        "test": df_test.reset_index(drop=True),
    }


def run(
    annotations_csv: Path = ANNOTATIONS_CSV,
    splits_dir: Path = SPLITS_DIR,
    cfg: Config | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    existing = [n for n in SPLIT_NAMES if (splits_dir / f"{n}.csv").exists()]
    if existing and not force:
        raise FileExistsError(
            f"split が既に存在します ({existing})。作り直す場合は --force を指定してください。"
            " 注意: 作り直すと test セットが変わり、これまでの評価と比較できなくなります。"
        )

    annotations = pd.read_csv(annotations_csv)
    if len(annotations) < 10:
        raise ValueError(f"アノテーションが少なすぎます ({len(annotations)}件)。最低10件必要です。")

    splits = make_splits(annotations, cfg)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(splits_dir / f"{name}.csv", index=False)
    sizes = {name: len(df) for name, df in splits.items()}
    print(f"split完了: {sizes} → {splits_dir}")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="既存splitを作り直す")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
