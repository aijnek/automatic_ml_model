"""Pixabay API (https://pixabay.com/api/docs/)。要無料APIキー。

ライセンスは Pixabay Content License（自由利用可・再配布に一部制限）。
CSVには license="pixabay" と記録する。レート制限は 100リクエスト/分。
"""

from __future__ import annotations

import os

from .base import ImageSource, SearchResult, http_get_json

API_URL = "https://pixabay.com/api/"


class PixabaySource(ImageSource):
    name = "pixabay"
    prefix = "px"
    search_sleep = 1.0
    env_vars = ("PIXABAY_API_KEY",)

    def search(self, query: str, page: int) -> list[SearchResult]:
        data = http_get_json(
            API_URL,
            params={
                "key": os.environ["PIXABAY_API_KEY"],
                "q": query,
                "image_type": "photo",
                "safesearch": "true",
                "per_page": 20,
                "page": page,
            },
        )
        results = []
        for hit in data.get("hits", []):
            if not hit.get("largeImageURL"):
                continue
            results.append(
                SearchResult(
                    source=self.name,
                    source_id=str(hit["id"]),
                    image_url=hit["largeImageURL"],  # 1280px 版
                    thumbnail_url=hit.get("webformatURL") or hit["largeImageURL"],
                    foreign_landing_url=hit.get("pageURL") or "",
                    license="pixabay",
                    creator=hit.get("user") or "",
                    tags=[t.strip() for t in (hit.get("tags") or "").split(",") if t.strip()],
                )
            )
        return results
