"""保育写真スコアリングモデル用の学習画像を各種画像APIから収集する。

売れる写真だけでなく「売れない写真」のバリエーション（ブレ・逆光・後ろ姿・
表情不明瞭・人物なし・人物極小・顔見切れ）を8カテゴリで集める。
検索で集まりにくいカテゴリは、追加取得した実写をベースに Pillow 加工で補完する。

使い方:
    uv run python scripts/collect_images.py                 # 8カテゴリ x 50枚
    uv run python scripts/collect_images.py --per-category 3
    uv run python scripts/collect_images.py --categories good,backlit
    uv run python scripts/collect_images.py --sources wikimedia,flickr

ソース: openverse, pixabay, pexels, wikimedia, flickr, google（scripts/sources/ 参照）。
APIキーが必要なソースはプロジェクトルートの .env（PIXABAY_API_KEY 等）で設定する。
--sources auto（既定）はキー設定済みの既定ソース全部を使う（google は除く）。

出力:
    data/images/{category}_{連番}_{ソース略号}_{id先頭8桁}.jpg  # 実写
    data/images/{category}_syn_{連番}.jpg                       # 加工画像
    data/collection_metadata.csv                                # 出典・ライセンス・加工パラメータ

再実行すると metadata.csv を読んで取得済み画像をスキップし、続きから収集する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import random
import shutil
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import DATA_DIR, IMAGES_DIR  # noqa: E402
from scripts.sources import get_sources, load_dotenv  # noqa: E402
from scripts.sources.base import (  # noqa: E402
    USER_AGENT,
    ImageSource,
    SearchResult,
    safe_id,
)

METADATA_CSV = DATA_DIR / "collection_metadata.csv"
DOWNLOAD_SLEEP = 0.7
MIN_SHORT_SIDE = 400
MAX_ASPECT = 3.0
MAX_PAGES_PER_QUERY = 5

METADATA_FIELDS = [
    "filename",
    "category",
    "is_synthetic",
    "source",
    "source_id",
    "base_filename",
    "image_url",
    "foreign_landing_url",
    "license",
    "license_version",
    "creator",
    "query",
    "augmentation_params",
]

# タイトル・タグの関連性フィルタ用キーワード
PERSON_WORDS = [
    "child", "children", "kid", "kids", "boy", "girl", "toddler", "baby",
    "preschool", "kindergarten", "school", "family", "people", "person",
    "man", "woman", "student", "pupil",
]
PLACE_WORDS = [
    "playground", "classroom", "school", "kindergarten", "swing", "slide",
    "gym", "nursery", "park",
]

# カテゴリ定義:
#   queries   実写の検索クエリ
#   keywords  タイトルまたはタグにいずれかを含まない画像は除外（関連性フィルタ）
#   synthetic 不足分の補完に使う加工名
CATEGORIES: dict[str, dict] = {
    "good": {
        "queries": [
            "kindergarten children playing",
            "preschool kids smiling",
            "happy child smiling portrait",
            "children sports day running race",
            "kids classroom activity drawing",
            "children playground happy",
        ],
        "keywords": PERSON_WORDS,
        "synthetic": None,
    },
    "motion_blur": {
        "queries": [
            "children running motion blur",
            "kids playing blurred motion",
            "child running fast blur",
            "motion blur people walking",
        ],
        "keywords": PERSON_WORDS,
        "synthetic": "motion_blur",
    },
    "backlit": {
        "queries": [
            "child silhouette backlit",
            "children playground sunset silhouette",
            "kids backlight shadow",
            "person silhouette sunset beach",
        ],
        "keywords": PERSON_WORDS + ["silhouette"],
        "synthetic": "backlit",
    },
    "back_view": {
        "queries": [
            "child from behind",
            "children back view walking",
            "kids rear view backpack",
            "child walking away",
            "people walking away back",
        ],
        "keywords": PERSON_WORDS,
        "synthetic": None,
    },
    "unclear_face": {
        "queries": [
            "child looking away candid",
            "children side profile",
            "child looking down playing",
            "kid candid playing absorbed",
        ],
        "keywords": PERSON_WORDS,
        "synthetic": "unclear_face",
    },
    "no_person": {
        "queries": [
            "empty playground",
            "empty kindergarten classroom",
            "playground equipment slide swing",
            "school gymnasium empty",
            "nursery school interior",
        ],
        "keywords": PLACE_WORDS,
        "synthetic": None,
    },
    "tiny_person": {
        "queries": [
            "children playing distance wide shot",
            "people walking park distance",
            "schoolyard children playing",
            "sports field game spectators",
            "beach people distance landscape",
        ],
        "keywords": PERSON_WORDS,
        "synthetic": None,
    },
    "cropped_face": {
        "queries": [
            "child face close up partial",
        ],
        "keywords": PERSON_WORDS,
        "synthetic": "cropped_face",
    },
}

# 加工のベース画像として追加取得するときに使うクエリ（good と重複しない子ども写真）
BASE_POOL_QUERIES = [
    "children playing outdoors",
    "kids having fun park",
    "child portrait outdoor",
    "children birthday party",
    "toddler playing garden",
]


def load_metadata() -> list[dict]:
    """メタデータCSVを読む。旧スキーマ（openverse_id 列）は自動で移行する。"""
    if not METADATA_CSV.exists():
        return []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "source" in fieldnames or "openverse_id" not in fieldnames:
        return rows
    # 旧スキーマ: openverse_id 列を source / source_id の2列に置き換える
    bak = METADATA_CSV.with_name(METADATA_CSV.name + ".bak")
    if not bak.exists():
        shutil.copy2(METADATA_CSV, bak)
    for row in rows:
        sid = row.pop("openverse_id", "") or ""
        row["source"] = "openverse" if sid else ""
        row["source_id"] = sid
    tmp = METADATA_CSV.with_name(METADATA_CSV.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, METADATA_CSV)
    print(f"{METADATA_CSV.name} を新スキーマへ移行しました（backup: {bak.name}）")
    return rows


def append_metadata(rows: list[dict]) -> None:
    exists = METADATA_CSV.exists()
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def is_relevant(result: SearchResult, keywords: list[str]) -> bool:
    """タイトルまたはタグに keywords のいずれかを含むか（無関係画像の除外用）。"""
    text = (result.title + " " + " ".join(result.tags)).lower()
    return any(k in text for k in keywords)


def download_image(url: str) -> Image.Image | None:
    """画像をダウンロードして検証済みの PIL Image を返す。不適格なら None。"""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        if len(resp.content) > 15 * 1024 * 1024:
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except Exception:
        return None
    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE or max(w, h) / min(w, h) > MAX_ASPECT:
        return None
    return img.convert("RGB")


class Collector:
    def __init__(self):
        self.rows: list[dict] = load_metadata()
        self.seen_ids = {
            f"{r['source']}:{r['source_id']}" for r in self.rows if r["source_id"]
        }
        # 掲載元URLでのソース横断重複排除（例: Openverse経由で取得済みの
        # Flickr写真を Flickr 直APIで再取得しない）
        self.seen_foreign = {
            r["foreign_landing_url"].rstrip("/")
            for r in self.rows
            if r["foreign_landing_url"]
        }
        self.seen_hashes: set[str] = set()
        for r in self.rows:
            path = IMAGES_DIR / r["filename"]
            if path.exists():
                self.seen_hashes.add(hashlib.md5(path.read_bytes()).hexdigest())

    def count(self, category: str) -> int:
        return sum(
            1
            for r in self.rows
            if r["category"] == category and (IMAGES_DIR / r["filename"]).exists()
        )

    def save(self, img: Image.Image, filename: str, meta: dict) -> None:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        img.save(IMAGES_DIR / filename, "JPEG", quality=90)
        row = {k: "" for k in METADATA_FIELDS} | meta | {"filename": filename}
        self.rows.append(row)
        append_metadata([row])

    def collect_from_queries(
        self,
        category: str,
        queries: list[str],
        target: int,
        keywords: list[str],
        sources: list[ImageSource],
    ) -> int:
        """target 枚に達するまで検索・ダウンロードする。取得できた枚数を返す。

        page -> query -> source の順で回してソースをインターリーブし、
        単一ソースへの偏りを防ぐ。レート制限はソースごとに throttle() で管理。
        """
        got = self.count(category)
        if got >= target:
            print(f"  [{category}] 既に {got} 枚あり。スキップ")
            return got
        for page in range(1, MAX_PAGES_PER_QUERY + 1):
            for query in queries:
                for source in sources:
                    if got >= target:
                        return got
                    source.throttle()
                    try:
                        results = source.search(query, page)
                    except Exception as e:
                        print(f"    検索失敗 ({source.name} {query} p{page}): {e}")
                        continue
                    for r in results:
                        if got >= target:
                            return got
                        if r.key in self.seen_ids or not r.image_url:
                            continue
                        foreign = r.foreign_landing_url.rstrip("/")
                        if foreign and foreign in self.seen_foreign:
                            continue
                        if not is_relevant(r, keywords):
                            continue
                        self.seen_ids.add(r.key)
                        img = download_image(r.image_url)
                        time.sleep(DOWNLOAD_SLEEP)
                        if img is None:
                            continue
                        buf = io.BytesIO()
                        img.save(buf, "JPEG", quality=90)
                        digest = hashlib.md5(buf.getvalue()).hexdigest()
                        if digest in self.seen_hashes:
                            continue
                        self.seen_hashes.add(digest)
                        if foreign:
                            self.seen_foreign.add(foreign)
                        got += 1
                        filename = (
                            f"{category}_{got:04d}_{source.prefix}_{safe_id(r.source_id)}.jpg"
                        )
                        self.save(
                            img,
                            filename,
                            {
                                "category": category,
                                "is_synthetic": "false",
                                "source": r.source,
                                "source_id": r.source_id,
                                "image_url": r.image_url,
                                "foreign_landing_url": r.foreign_landing_url,
                                "license": r.license,
                                "license_version": r.license_version,
                                "creator": r.creator,
                                "query": query,
                            },
                        )
                        print(f"  [{category}] {got}/{target} {filename}")
        return got


# ---------------------------------------------------------------- 加工（合成）

def aug_motion_blur(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    passes = rng.randint(2, 4)
    kernel = [0.0] * 25
    for i in range(5):  # 水平方向の 5x5 モーションカーネル
        kernel[2 * 5 + i] = 1 / 5
    out = img
    for _ in range(passes):
        out = out.filter(ImageFilter.Kernel((5, 5), kernel))
    return out, f"motion_blur:passes={passes}"

def aug_backlit(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    brightness = rng.uniform(0.3, 0.5)
    contrast = rng.uniform(1.1, 1.4)
    out = ImageEnhance.Brightness(img).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    gamma = rng.uniform(1.3, 1.8)  # 暗部をさらに潰す
    lut = [int(255 * (i / 255) ** gamma) for i in range(256)] * 3
    out = out.point(lut)
    return out, f"backlit:brightness={brightness:.2f},contrast={contrast:.2f},gamma={gamma:.2f}"

def aug_cropped_face(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    # 顔検出なしの近似: 画像の端に寄せた強めのクロップで被写体を見切れさせる
    w, h = img.size
    keep = rng.uniform(0.45, 0.6)
    cw, ch = int(w * keep), int(h * keep)
    corner = rng.choice(["left", "right", "top", "bottom"])
    if corner == "left":
        box = (0, (h - ch) // 2, cw, (h - ch) // 2 + ch)
    elif corner == "right":
        box = (w - cw, (h - ch) // 2, w, (h - ch) // 2 + ch)
    elif corner == "top":
        box = ((w - cw) // 2, 0, (w - cw) // 2 + cw, ch)
    else:
        box = ((w - cw) // 2, h - ch, (w - cw) // 2 + cw, h)
    return img.crop(box), f"cropped_face:keep={keep:.2f},edge={corner}"

def aug_unclear_face(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    scale = rng.uniform(0.2, 0.35)
    w, h = img.size
    out = img.resize((int(w * scale), int(h * scale))).resize((w, h))
    radius = rng.uniform(1.5, 3.0)
    out = out.filter(ImageFilter.GaussianBlur(radius))
    return out, f"unclear_face:scale={scale:.2f},blur={radius:.1f}"

AUGMENTATIONS = {
    "motion_blur": aug_motion_blur,
    "backlit": aug_backlit,
    "cropped_face": aug_cropped_face,
    "unclear_face": aug_unclear_face,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument(
        "--categories",
        default=",".join(CATEGORIES),
        help="カンマ区切りで対象カテゴリを限定",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        default="auto",
        help="カンマ区切り (openverse,pixabay,pexels,wikimedia,flickr,google)。"
        "auto=キー設定済みの既定ソース全部（google を除く）",
    )
    args = parser.parse_args()

    targets = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = set(targets) - set(CATEGORIES)
    if unknown:
        parser.error(f"未知のカテゴリ: {unknown}")

    load_dotenv()
    names = (
        None
        if args.sources == "auto"
        else [s.strip() for s in args.sources.split(",") if s.strip()]
    )
    try:
        sources = get_sources(names)
    except ValueError as e:
        parser.error(str(e))
    if not sources:
        parser.error("使用可能なソースがありません（.env のAPIキーまたは --sources を確認）")
    print(f"使用ソース: {', '.join(s.name for s in sources)}")

    rng = random.Random(args.seed)
    collector = Collector()

    # Phase 1: 実写の収集
    print("=== Phase 1: 実写収集 ===")
    for category in targets:
        print(f"[{category}] 目標 {args.per_category} 枚")
        collector.collect_from_queries(
            category,
            CATEGORIES[category]["queries"],
            args.per_category,
            CATEGORIES[category]["keywords"],
            sources,
        )

    # Phase 2: 不足カテゴリを加工で補完
    shortfalls = {
        c: args.per_category - collector.count(c)
        for c in targets
        if CATEGORIES[c]["synthetic"] and collector.count(c) < args.per_category
    }
    if shortfalls:
        total_needed = sum(shortfalls.values())
        print(f"=== Phase 2: 加工補完（{shortfalls} / ベース画像 {total_needed} 枚追加取得）===")
        # 既存カテゴリと重複しない画像をベースとして追加取得（split 間の近重複リークを防ぐ）
        collector.collect_from_queries(
            "_base_pool", BASE_POOL_QUERIES, total_needed, PERSON_WORDS, sources
        )
        base_rows = [r for r in collector.rows if r["category"] == "_base_pool"]
        rng.shuffle(base_rows)
        idx = 0
        for category, needed in shortfalls.items():
            aug_fn = AUGMENTATIONS[CATEGORIES[category]["synthetic"]]
            for i in range(needed):
                if idx >= len(base_rows):
                    print(f"  [{category}] ベース画像不足。{i}/{needed} 枚で終了")
                    break
                base = base_rows[idx]
                idx += 1
                img = Image.open(IMAGES_DIR / base["filename"]).convert("RGB")
                out, params = aug_fn(img, rng)
                n = collector.count(category) + 1
                filename = f"{category}_syn_{n:04d}.jpg"
                collector.save(
                    out,
                    filename,
                    {
                        "category": category,
                        "is_synthetic": "true",
                        "base_filename": base["filename"],
                        "image_url": base["image_url"],
                        "foreign_landing_url": base["foreign_landing_url"],
                        "license": base["license"],
                        "license_version": base["license_version"],
                        "creator": base["creator"],
                        "augmentation_params": params,
                    },
                )
                print(f"  [{category}] 加工 {filename} <- {base['filename']}")
        # ベース画像はデータセット本体には含めない（加工版のみ残す）
        for base in base_rows:
            (IMAGES_DIR / base["filename"]).unlink(missing_ok=True)
        collector.rows = [r for r in collector.rows if r["category"] != "_base_pool"]
        with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            writer.writerows(collector.rows)

    # サマリ
    print("=== 収集結果 ===")
    total = 0
    for category in targets:
        rows = [
            r
            for r in collector.rows
            if r["category"] == category and (IMAGES_DIR / r["filename"]).exists()
        ]
        real = sum(1 for r in rows if r["is_synthetic"] == "false")
        syn = len(rows) - real
        total += len(rows)
        print(f"  {category:14s} 計{len(rows):4d} 枚（実写 {real} / 加工 {syn}）")
    print(f"  合計 {total} 枚 -> {IMAGES_DIR}")
    print(f"  メタデータ: {METADATA_CSV}")


if __name__ == "__main__":
    main()
