"""Wikimedia Commons MediaWiki API。APIキー不要（記述的User-Agent必須）。

ライセンスは画像ごとに extmetadata から取得する（例 "CC BY-SA 4.0"）。
原本は100MB超のTIFF等がありうるため、1600px のサムネイルレンダを取得する。
"""

from __future__ import annotations

import re

from .base import ImageSource, SearchResult, http_get_json

API_URL = "https://commons.wikimedia.org/w/api.php"
OK_MIMES = {"image/jpeg", "image/png"}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


class WikimediaSource(ImageSource):
    name = "wikimedia"
    prefix = "wm"
    search_sleep = 1.5

    def search(self, query: str, page: int) -> list[SearchResult]:
        data = http_get_json(
            API_URL,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrnamespace": 6,  # File 名前空間
                "gsrlimit": 20,
                "gsroffset": (page - 1) * 20,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": 1600,
            },
        )
        results = []
        for p in (data.get("query", {}).get("pages") or {}).values():
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") not in OK_MIMES or not info.get("thumburl"):
                continue
            ext = info.get("extmetadata") or {}

            def meta(key: str) -> str:
                return str((ext.get(key) or {}).get("value") or "")

            title = re.sub(r"^File:", "", p.get("title") or "")
            # 説明文もタイトルに足して関連性フィルタに使えるようにする
            description = _strip_html(meta("ImageDescription"))
            results.append(
                SearchResult(
                    source=self.name,
                    source_id=str(p["pageid"]),
                    image_url=info["thumburl"],
                    thumbnail_url=info["thumburl"],
                    foreign_landing_url=info.get("descriptionurl") or "",
                    license=meta("LicenseShortName"),
                    creator=_strip_html(meta("Artist")),
                    title=f"{title} {description}".strip(),
                )
            )
        return results
