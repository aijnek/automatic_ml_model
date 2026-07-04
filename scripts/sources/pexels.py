"""Pexels API (https://www.pexels.com/api/documentation/)。要無料APIキー。

ライセンスは Pexels License（自由利用可）。CSVには license="pexels" と記録する。
レート制限は 200リクエスト/時（検索のみ。画像CDNからのダウンロードは対象外）
のため検索間隔を 20 秒に設定している。
"""

from __future__ import annotations

import os

from .base import ImageSource, SearchResult, http_get_json

API_URL = "https://api.pexels.com/v1/search"


class PexelsSource(ImageSource):
    name = "pexels"
    prefix = "pe"
    search_sleep = 20.0
    env_vars = ("PEXELS_API_KEY",)

    def search(self, query: str, page: int) -> list[SearchResult]:
        data = http_get_json(
            API_URL,
            params={"query": query, "per_page": 20, "page": page},
            headers={"Authorization": os.environ["PEXELS_API_KEY"]},
        )
        results = []
        for p in data.get("photos", []):
            src = p.get("src") or {}
            # original は 15MB を超えることがあるので large2x（~2560px）を優先
            image_url = src.get("large2x") or src.get("original")
            if not image_url:
                continue
            results.append(
                SearchResult(
                    source=self.name,
                    source_id=str(p["id"]),
                    image_url=image_url,
                    thumbnail_url=src.get("medium") or image_url,
                    foreign_landing_url=p.get("url") or "",
                    license="pexels",
                    creator=p.get("photographer") or "",
                    title=p.get("alt") or "",
                )
            )
        return results
