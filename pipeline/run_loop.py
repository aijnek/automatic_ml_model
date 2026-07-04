"""自律改善ループのオーケストレータ（step 4〜8）。

実行: uv run python -m pipeline.run_loop
以降は完全無人で「特徴量設計→抽出→学習→評価」を繰り返し、
val スコアが合格ラインを超えたら test で最終評価して終了する。

- 各フェーズの成果物（スキーマ/特徴量CSV/モデル/レポート）は毎回ディスクに保存され、
  Ctrl+C や障害で落ちても同コマンドで途中から再開できる
- スキーマファイルが既に存在するイテレーションは designer をスキップ
- 特徴量CSVは1画像ごとに追記されるため抽出も途中から再開される
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from pipeline.config import (
    FEATURES_DIR,
    IMAGES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SCHEMAS_DIR,
    SPLITS_DIR,
    STATE_PATH,
    Config,
    load_config,
)
from pipeline.designer import active_features, design_schema
from pipeline.extract import run_extraction
from pipeline.report import (
    build_final_report_md,
    save_iteration_report,
)
from pipeline.train import evaluate_on_test, train_and_evaluate

logger = logging.getLogger("run_loop")


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"iteration": 1, "best": None, "history": [], "finished": False}


def save_state(state: dict, state_path: Path) -> None:
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def _setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(results_dir / "loop.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} がありません。先に `uv run python -m pipeline.split` を実行してください。"
            )
        splits[name] = pd.read_csv(path)
    return splits


def _get_schema(
    cfg: Config,
    iteration: int,
    schemas_dir: Path,
    results_dir: Path,
    designer_llm_fn: Callable[[str], str] | None,
) -> tuple[dict, dict | None]:
    """今回のスキーマと前回スキーマを返す。既存ファイルがあれば designer をスキップ。"""
    prev_schema = None
    if iteration > 1:
        prev_schema = json.loads(
            (schemas_dir / f"metadata_v{iteration - 1}.json").read_text()
        )

    schema_path = schemas_dir / f"metadata_v{iteration}.json"
    if schema_path.exists():
        logger.info("iter %d: 既存スキーマ %s を使用（designerスキップ）", iteration, schema_path.name)
        return json.loads(schema_path.read_text()), prev_schema

    prev_report_md = None
    if iteration > 1:
        report_path = results_dir / f"iter{iteration - 1}" / "report.md"
        prev_report_md = report_path.read_text() if report_path.exists() else None

    logger.info("iter %d: designer で特徴量スキーマを%s中...", iteration, "改訂" if prev_schema else "設計")
    schema = design_schema(
        cfg,
        version=iteration,
        prev_schema=prev_schema,
        prev_report_md=prev_report_md,
        llm_fn=designer_llm_fn,
    )
    schemas_dir.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2))
    changes = {a: sum(1 for f in schema["features"] if f.get("action") == a)
               for a in ("new", "modified", "kept", "removed")}
    logger.info("iter %d: スキーマ確定 (有効特徴量 %d個, 変更内訳 %s)",
                iteration, len(active_features(schema)), changes)
    return schema, prev_schema


def run(
    cfg: Config | None = None,
    splits_dir: Path = SPLITS_DIR,
    images_dir: Path = IMAGES_DIR,
    schemas_dir: Path = SCHEMAS_DIR,
    features_dir: Path = FEATURES_DIR,
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    state_path: Path = STATE_PATH,
    max_iterations: int | None = None,
    designer_llm_fn: Callable[[str], str] | None = None,
    vlm_fn: Callable[..., str] | None = None,
) -> dict:
    """自律ループ本体。designer_llm_fn / vlm_fn はテスト用の差し替え口。"""
    cfg = cfg or load_config()
    _setup_logging(results_dir)
    splits = _load_splits(splits_dir)
    all_filenames = pd.concat([s["filename"] for s in splits.values()]).tolist()

    state = load_state(state_path)
    if state["finished"]:
        logger.info("既に完了しています（final_report.md 参照）。やり直す場合は state.json を削除してください。")
        return state

    while True:
        iteration = state["iteration"]
        logger.info("===== イテレーション %d 開始 =====", iteration)

        # step 4: 特徴量スキーマの設計/改訂
        schema, prev_schema = _get_schema(
            cfg, iteration, schemas_dir, results_dir, designer_llm_fn
        )

        # step 5: メタデータ抽出（差分・再開対応）
        logger.info("iter %d: VLM (%s) でメタデータ抽出中...", iteration, cfg.vlm_model)
        features_df = run_extraction(
            schema,
            all_filenames,
            images_dir,
            features_dir / f"features_v{iteration}.csv",
            cfg.vlm_model,
            prev_schema=prev_schema,
            prev_csv=features_dir / f"features_v{iteration - 1}.csv",
            vlm_fn=vlm_fn,
        )

        # step 6: LightGBM 学習・val 評価
        result = train_and_evaluate(
            cfg,
            schema,
            features_df,
            splits,
            model_path=models_dir / f"model_v{iteration}.txt",
        )
        save_iteration_report(cfg, iteration, result, results_dir / f"iter{iteration}")

        state["history"].append(
            {
                "iteration": iteration,
                "val_score": result["val_score"],
                "n_features": len(active_features(schema)),
            }
        )
        if state["best"] is None or result["val_score"] > state["best"]["val_score"]:
            state["best"] = {"iteration": iteration, "val_score": result["val_score"]}

        # step 7: 合格判定
        passed = result["val_score"] >= cfg.threshold
        logger.info(
            "iter %d: val %s = %.4f（合格ライン %.2f → %s）",
            iteration,
            cfg.metric_name,
            result["val_score"],
            cfg.threshold,
            "合格" if passed else "不合格",
        )

        if passed:
            # step 8: test で最終評価（1回のみ）
            test_result = evaluate_on_test(
                cfg, schema, features_df, splits["test"], result["_model"]
            )
            final_md = build_final_report_md(
                cfg, iteration, state["history"], test_result, schema
            )
            (results_dir / "final_report.md").write_text(final_md)
            (results_dir / "final_report.json").write_text(
                json.dumps(test_result, ensure_ascii=False, indent=2)
            )
            state["finished"] = True
            state["final_iteration"] = iteration
            state["test_score"] = test_result["test_score"]
            save_state(state, state_path)
            logger.info(
                "🎉 完了: test %s = %.4f → %s",
                cfg.metric_name,
                test_result["test_score"],
                results_dir / "final_report.md",
            )
            return state

        state["iteration"] = iteration + 1
        save_state(state, state_path)

        if max_iterations is not None and iteration >= max_iterations:
            logger.info("max_iterations (%d) に到達。中断します（再実行で継続）。", max_iterations)
            return state


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="このイテレーション数で一旦停止する（既定: 合格まで無制限）",
    )
    args = parser.parse_args()
    run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
