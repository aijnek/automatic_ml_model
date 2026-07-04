"""画像検索結果を一覧し、選んだ画像だけをデータセットに追加する収集UI。

使い方:
    uv run streamlit run app/collect.py

サイドバーで検索ソース（Openverse / Pixabay / Pexels / Wikimedia / Flickr / Google）
とカテゴリを選び、クエリを打って検索 → サムネイル一覧から適した画像だけチェック → 保存。
APIキーが必要なソースはプロジェクトルートの .env（PIXABAY_API_KEY 等）で設定する。
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
)
from scripts.sources import SOURCES, load_dotenv  # noqa: E402
from scripts.sources.base import SearchResult, safe_id  # noqa: E402

st.set_page_config(page_title="画像収集", layout="wide")

GRID_COLS = 4

load_dotenv()


def dataset_index() -> tuple[dict[str, dict], dict[str, int]]:
    """メタデータを読み、キー -> 状態（saved/excluded）とカテゴリ別枚数を返す。

    キーは "source:source_id" と掲載元URL（rstrip("/")）の両方で引ける
    （ソース横断の重複検出のため。例: Openverse経由で保存済みのFlickr写真）。
    """
    status: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for row in load_metadata():
        exists = (IMAGES_DIR / row["filename"]).exists()
        entry = {"exists": exists, "row": row}
        if row["source_id"]:
            status[f"{row['source']}:{row['source_id']}"] = entry
        if row["foreign_landing_url"]:
            status[row["foreign_landing_url"].rstrip("/")] = entry
        if exists:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
    return status, counts


def lookup_status(result: SearchResult, status: dict[str, dict]) -> dict | None:
    known = status.get(result.key)
    if known is None and result.foreign_landing_url:
        known = status.get(result.foreign_landing_url.rstrip("/"))
    return known


def next_seq(category: str) -> int:
    """カテゴリ内の既存ファイルの最大連番 + 1（削除済み分の欠番は再利用しない）。"""
    pat = re.compile(rf"^{re.escape(category)}_(\d{{4}})_")
    seqs = [
        int(m.group(1))
        for p in IMAGES_DIR.glob(f"{category}_*.jpg")
        if (m := pat.match(p.name))
    ]
    return max(seqs, default=0) + 1


def license_label(result: SearchResult) -> str:
    """ライセンス表示。CCスラッグ（by-nc-nd等）だけ CC を冠する。"""
    lic = result.license or "?"
    if " " in lic or lic in ("pixabay", "pexels", "unknown", "?"):
        display = lic
    else:
        display = f"CC {lic.upper()}"
    return f"{display} / {result.creator or '不明'}"


def save_result(result: SearchResult, category: str, query: str) -> str | None:
    """検索結果1件をフル解像度で取得して保存する。失敗理由を返す（成功なら None）。"""
    img = download_image(result.image_url)
    if img is None:
        return "ダウンロード失敗または画質基準未満（短辺400px未満など）"
    prefix = SOURCES[result.source].prefix
    filename = (
        f"{category}_{next_seq(category):04d}_{prefix}_{safe_id(result.source_id)}.jpg"
    )
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img.save(IMAGES_DIR / filename, "JPEG", quality=90)
    append_metadata(
        [
            {k: "" for k in METADATA_FIELDS}
            | {
                "filename": filename,
                "category": category,
                "is_synthetic": "false",
                "source": result.source,
                "source_id": result.source_id,
                "image_url": result.image_url,
                "foreign_landing_url": result.foreign_landing_url,
                "license": result.license,
                "license_version": result.license_version,
                "creator": result.creator,
                "query": query,
            }
        ]
    )
    return None


def render_card(result: SearchResult, status: dict[str, dict], hide_dups: bool) -> None:
    known = lookup_status(result, status)
    license_line = license_label(result)
    if known:
        if hide_dups:
            return
        label = "✅ 保存済み" if known["exists"] else "🚫 除外済み（過去に削除）"
        st.markdown(
            f'<img src="{result.thumbnail_url}" style="opacity:0.3;width:100%;border-radius:4px">',
            unsafe_allow_html=True,
        )
        st.caption(f"{label}\n\n{license_line}")
    else:
        st.image(result.thumbnail_url, use_container_width=True)
        st.checkbox("データセットに入れる", key=f"sel_{result.key}")
        st.caption(f"[{result.title or '(無題)'}]({result.foreign_landing_url})\n\n{license_line}")


def main() -> None:
    st.title("🖼️ 画像収集")
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
        usable = [s for s in SOURCES.values() if s.available()[0]]
        source = st.selectbox("検索ソース", usable, format_func=lambda s: s.name)
        for s in SOURCES.values():
            ok, reason = s.available()
            if not ok:
                st.caption(f"{s.name}: {reason}")
        if source.name == "google":
            st.caption("⚠️ ライセンス情報が取得できないため license=unknown で記録されます")
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
        with st.spinner(f"{source.name} を検索中..."):
            st.session_state.results = source.search(query, 1)
            st.session_state.page = 1
            st.session_state.last_query = query
            st.session_state.source_name = source.name
        if not st.session_state.results:
            st.warning("ヒットなし。クエリを変えて再検索してください。")
    if "results" in st.session_state and col_more.button("⬇️ さらに20件"):
        with st.spinner("追加取得中..."):
            st.session_state.page += 1
            more = SOURCES[st.session_state.source_name].search(
                st.session_state.last_query, st.session_state.page
            )
        if more:
            seen = {r.key for r in st.session_state.results}
            st.session_state.results += [r for r in more if r.key not in seen]
        else:
            st.info("これ以上結果がありません。")

    results = st.session_state.get("results", [])
    if not results:
        st.info("クエリを入力して検索してください。良い画像がなければクエリを修正して再検索を繰り返せます。")
        return

    selected = [r for r in results if st.session_state.get(f"sel_{r.key}")]
    st.write(
        f"検索結果 {len(results)} 件"
        f"（ソース: `{st.session_state.source_name}` / クエリ: `{st.session_state.last_query}`）— "
        f"選択中 **{len(selected)} 枚** → 保存先カテゴリ **{category}**"
    )
    if st.button(f"💾 選択した {len(selected)} 枚を保存", disabled=not selected):
        progress = st.progress(0.0)
        errors = []
        for i, r in enumerate(selected):
            err = save_result(r, category, st.session_state.last_query)
            if err:
                errors.append(f"{r.title or r.source_id}: {err}")
            st.session_state[f"sel_{r.key}"] = False
            progress.progress((i + 1) / len(selected))
        st.session_state.save_report = (len(selected) - len(errors), category, errors)
        st.rerun()

    hide_dups = dup_mode == "非表示"
    cols = st.columns(GRID_COLS)
    for i, r in enumerate(results):
        with cols[i % GRID_COLS]:
            render_card(r, status, hide_dups)


main()
