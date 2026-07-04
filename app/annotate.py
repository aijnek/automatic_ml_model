"""問題設定 + 画像アノテーション用 Streamlit アプリ。

起動: uv run streamlit run app/annotate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (  # noqa: E402
    ANNOTATIONS_CSV,
    CONFIG_PATH,
    IMAGES_DIR,
    Config,
    load_config,
    save_config,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

st.set_page_config(page_title="ML Pipeline Annotator", layout="wide")


def load_annotations() -> dict[str, object]:
    if not ANNOTATIONS_CSV.exists():
        return {}
    df = pd.read_csv(ANNOTATIONS_CSV)
    return dict(zip(df["filename"].astype(str), df["label"]))


def save_annotations(annotations: dict[str, object]) -> None:
    ANNOTATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        sorted(annotations.items()), columns=["filename", "label"]
    )
    df.to_csv(ANNOTATIONS_CSV, index=False)


def list_images() -> list[str]:
    if not IMAGES_DIR.exists():
        return []
    return sorted(
        p.name for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )


def page_setup() -> None:
    st.header("1. 問題設定")
    existing: Config | None = None
    if CONFIG_PATH.exists():
        try:
            existing = load_config()
            st.info("既存の config.yaml を読み込みました。保存すると上書きされます。")
        except Exception as e:  # 壊れたconfigでも設定し直せるようにする
            st.warning(f"config.yaml の読み込みに失敗: {e}")

    task_type = st.radio(
        "タスク種別",
        ["classification", "regression"],
        index=0 if existing is None or existing.is_classification else 1,
        horizontal=True,
    )
    description = st.text_area(
        "タスク説明（何の画像で、何を予測したいか。特徴量設計LLMへの指示に使われるため具体的に）",
        value=existing.description if existing else "",
        height=120,
        placeholder="例: 保育園で撮影された園児の写真。販売に適した写真かどうかを分類したい。",
    )

    classes: list[str] = []
    target_min = target_max = None
    target_unit = ""
    if task_type == "classification":
        classes_text = st.text_input(
            "クラス名（カンマ区切り）",
            value=",".join(existing.classes) if existing else "",
            placeholder="例: optimal,neutral,unsuitable",
        )
        classes = [c.strip() for c in classes_text.split(",") if c.strip()]
    else:
        col1, col2, col3 = st.columns(3)
        target_min = col1.number_input(
            "ターゲット最小値",
            value=float(existing.target_min) if existing and existing.target_min is not None else 0.0,
        )
        target_max = col2.number_input(
            "ターゲット最大値",
            value=float(existing.target_max) if existing and existing.target_max is not None else 100.0,
        )
        target_unit = col3.text_input(
            "単位（任意）", value=existing.target_unit if existing else ""
        )

    threshold = st.slider(
        "合格ライン（分類: val macro-F1 / 回帰: val R²）",
        min_value=0.5,
        max_value=1.0,
        value=existing.threshold if existing else 0.9,
        step=0.01,
    )

    if st.button("設定を保存", type="primary"):
        cfg = Config(
            task_type=task_type,
            description=description,
            classes=classes,
            target_min=target_min,
            target_max=target_max,
            target_unit=target_unit,
            threshold=threshold,
        )
        if existing is not None:
            cfg.seed = existing.seed
            cfg.split_ratios = existing.split_ratios
            cfg.designer_command = existing.designer_command
            cfg.vlm_model = existing.vlm_model
        try:
            save_config(cfg)
            st.success(f"config.yaml に保存しました: {CONFIG_PATH}")
        except ValueError as e:
            st.error(str(e))


def annotation_widget(cfg: Config, filename: str, current: object | None) -> object | None:
    """ラベル入力UIを表示し、確定されたラベルを返す（未確定はNone）。"""
    if cfg.is_classification:
        st.caption(f"現在のラベル: **{current}**" if current is not None else "未アノテーション")
        cols = st.columns(len(cfg.classes))
        for col, cls in zip(cols, cfg.classes):
            if col.button(cls, key=f"label_{filename}_{cls}", use_container_width=True):
                return cls
        return None
    default = float(current) if current is not None else float(cfg.target_min or 0.0)
    value = st.number_input(
        f"ターゲット値 {f'({cfg.target_unit})' if cfg.target_unit else ''}",
        min_value=float(cfg.target_min) if cfg.target_min is not None else None,
        max_value=float(cfg.target_max) if cfg.target_max is not None else None,
        value=default,
        key=f"value_{filename}",
    )
    if st.button("このラベルで保存", type="primary", key=f"save_{filename}"):
        return value
    return None


def page_annotate() -> None:
    st.header("2. アノテーション")
    if not CONFIG_PATH.exists():
        st.error("先に「問題設定」ページで設定を保存してください。")
        return
    cfg = load_config()

    uploaded = st.file_uploader(
        "画像をアップロード（複数可）",
        type=[e.lstrip(".") for e in IMAGE_EXTS],
        accept_multiple_files=True,
    )
    if uploaded:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        new_count = 0
        for f in uploaded:
            dest = IMAGES_DIR / Path(f.name).name
            if not dest.exists():
                dest.write_bytes(f.getbuffer())
                new_count += 1
        if new_count:
            st.success(f"{new_count} 枚を保存しました → {IMAGES_DIR}")

    images = list_images()
    if not images:
        st.info("画像がまだありません。アップロードしてください。")
        return

    annotations = load_annotations()
    done = [f for f in images if f in annotations]
    pending = [f for f in images if f not in annotations]
    st.progress(len(done) / len(images), text=f"進捗: {len(done)} / {len(images)} 枚")

    mode = st.radio("表示対象", ["未アノテーションのみ", "すべて（修正用）"], horizontal=True)
    targets = pending if mode == "未アノテーションのみ" else images
    if not targets:
        st.success("すべての画像にアノテーション済みです 🎉 次は split を実行してください: "
                   "`uv run python -m pipeline.split`")
        return

    idx_key = "annot_idx"
    st.session_state.setdefault(idx_key, 0)
    st.session_state[idx_key] = min(st.session_state[idx_key], len(targets) - 1)

    col_prev, col_pos, col_next = st.columns([1, 3, 1])
    if col_prev.button("← 前へ"):
        st.session_state[idx_key] = max(0, st.session_state[idx_key] - 1)
    if col_next.button("次へ →"):
        st.session_state[idx_key] = min(len(targets) - 1, st.session_state[idx_key] + 1)
    filename = targets[st.session_state[idx_key]]
    col_pos.markdown(
        f"<div style='text-align:center'>{st.session_state[idx_key] + 1} / {len(targets)}: "
        f"<code>{filename}</code></div>",
        unsafe_allow_html=True,
    )

    col_img, col_label = st.columns([2, 1])
    col_img.image(str(IMAGES_DIR / filename), use_container_width=True)
    with col_label:
        label = annotation_widget(cfg, filename, annotations.get(filename))
        if label is not None:
            annotations[filename] = label
            save_annotations(annotations)
            # 未アノテーションのみ表示中はリストが縮むのでindexは据え置きで次が出る
            st.rerun()


page = st.sidebar.radio("ページ", ["問題設定", "アノテーション"])
if page == "問題設定":
    page_setup()
else:
    page_annotate()
