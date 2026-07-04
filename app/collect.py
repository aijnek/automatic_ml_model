"""Openverse 検索結果を一覧し、選んだ画像だけをデータセットに追加する収集UI。

使い方:
    uv run streamlit run app/collect.py

クエリを打って検索 → サムネイル一覧から適した画像だけチェック → 保存。
保存済み・除外済み（過去に削除した）画像はグレーアウト表示または非表示になる。
保存先は data/images/、出典・ライセンスは data/collection_metadata.csv に記録される
（scripts/collect_images.py と同じ形式・同じ重複管理）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import IMAGES_DIR  # noqa: E402
from scripts.collect_images import (  # noqa: E402
    CATEGORIES,
    METADATA_FIELDS,
    append_metadata,
    download_image,
    load_metadata,
    search_openverse,
)

st.set_page_config(page_title="画像収集", layout="wide")

GRID_COLS = 4


def dataset_index() -> tuple[dict[str, dict], dict[str, int]]:
    """メタデータを読み、id -> 状態（saved/excluded）とカテゴリ別枚数を返す。"""
    status: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for row in load_metadata():
        exists = (IMAGES_DIR / row["filename"]).exists()
        if row["openverse_id"]:
            status[row["openverse_id"]] = {"exists": exists, "row": row}
        if exists:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
    return status, counts


def next_seq(category: str) -> int:
    """カテゴリ内の既存ファイルの最大連番 + 1（削除済み分の欠番は再利用しない）。"""
    pat = re.compile(rf"^{re.escape(category)}_(\d{{4}})_")
    seqs = [
        int(m.group(1))
        for p in IMAGES_DIR.glob(f"{category}_*.jpg")
        if (m := pat.match(p.name))
    ]
    return max(seqs, default=0) + 1


def save_result(result: dict, category: str, query: str) -> str | None:
    """検索結果1件をフル解像度で取得して保存する。失敗理由を返す（成功なら None）。"""
    img = download_image(result["url"])
    if img is None:
        return "ダウンロード失敗または画質基準未満（短辺400px未満など）"
    filename = f"{category}_{next_seq(category):04d}_{result['id'][:8]}.jpg"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img.save(IMAGES_DIR / filename, "JPEG", quality=90)
    append_metadata(
        [
            {k: "" for k in METADATA_FIELDS}
            | {
                "filename": filename,
                "category": category,
                "is_synthetic": "false",
                "openverse_id": result["id"],
                "image_url": result["url"],
                "foreign_landing_url": result.get("foreign_landing_url", ""),
                "license": result.get("license", ""),
                "license_version": result.get("license_version", ""),
                "creator": result.get("creator", ""),
                "query": query,
            }
        ]
    )
    return None


def render_card(result: dict, status: dict[str, dict], hide_dups: bool) -> None:
    known = status.get(result["id"])
    license_line = f"CC {result.get('license', '?').upper()} / {result.get('creator') or '不明'}"
    if known:
        if hide_dups:
            return
        label = "✅ 保存済み" if known["exists"] else "🚫 除外済み（過去に削除）"
        st.markdown(
            f'<img src="{result["thumbnail"]}" style="opacity:0.3;width:100%;border-radius:4px">',
            unsafe_allow_html=True,
        )
        st.caption(f"{label}\n\n{license_line}")
    else:
        st.image(result["thumbnail"], use_container_width=True)
        st.checkbox("データセットに入れる", key=f"sel_{result['id']}")
        st.caption(f"[{result.get('title') or '(無題)'}]({result.get('foreign_landing_url', '')})\n\n{license_line}")


def main() -> None:
    st.title("🖼️ 画像収集（Openverse）")
    if report := st.session_state.pop("save_report", None):
        ok, saved_cat, errors = report
        if ok:
            st.success(f"{ok} 枚を {saved_cat} に保存しました")
        for e in errors:
            st.warning(e)
    status, counts = dataset_index()

    with st.sidebar:
        st.header("設定")
        category = st.selectbox("保存先カテゴリ", list(CATEGORIES))
        dup_mode = st.radio("保存済み・除外済みの画像", ["グレーアウト表示", "非表示"])
        st.divider()
        st.subheader("現在のデータセット")
        for cat in CATEGORIES:
            st.text(f"{cat:14s} {counts.get(cat, 0):4d} 枚")
        st.text(f"{'合計':14s} {sum(counts.values()):4d} 枚")
        st.caption("クエリ例（カテゴリの既定クエリ）:")
        for q in CATEGORIES[category]["queries"]:
            st.caption(f"・{q}")

    query = st.text_input(
        "検索クエリ（英語が最もヒットしやすい）",
        value=CATEGORIES[category]["queries"][0],
        key="query_input",
    )
    col_search, col_more, _ = st.columns([1, 1, 3])
    if col_search.button("🔍 検索", type="primary"):
        with st.spinner("検索中..."):
            st.session_state.results = search_openverse(query, 1)
            st.session_state.page = 1
            st.session_state.last_query = query
        if not st.session_state.results:
            st.warning("ヒットなし。クエリを変えて再検索してください。")
    if "results" in st.session_state and col_more.button("⬇️ さらに20件"):
        with st.spinner("追加取得中..."):
            st.session_state.page += 1
            more = search_openverse(st.session_state.last_query, st.session_state.page)
        if more:
            seen = {r["id"] for r in st.session_state.results}
            st.session_state.results += [r for r in more if r["id"] not in seen]
        else:
            st.info("これ以上結果がありません。")

    results = st.session_state.get("results", [])
    if not results:
        st.info("クエリを入力して検索してください。良い画像がなければクエリを修正して再検索を繰り返せます。")
        return

    selected = [r for r in results if st.session_state.get(f"sel_{r['id']}")]
    st.write(
        f"検索結果 {len(results)} 件（クエリ: `{st.session_state.last_query}`）— "
        f"選択中 **{len(selected)} 枚** → 保存先カテゴリ **{category}**"
    )
    if st.button(f"💾 選択した {len(selected)} 枚を保存", disabled=not selected):
        progress = st.progress(0.0)
        errors = []
        for i, r in enumerate(selected):
            err = save_result(r, category, st.session_state.last_query)
            if err:
                errors.append(f"{r.get('title') or r['id']}: {err}")
            st.session_state[f"sel_{r['id']}"] = False
            progress.progress((i + 1) / len(selected))
        st.session_state.save_report = (len(selected) - len(errors), category, errors)
        st.rerun()

    hide_dups = dup_mode == "非表示"
    cols = st.columns(GRID_COLS)
    for i, r in enumerate(results):
        with cols[i % GRID_COLS]:
            render_card(r, status, hide_dups)


main()
