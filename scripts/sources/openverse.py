"""Openverse API (https://api.openverse.org/)。APIキー不要。CCライセンス画像。"""

from __future__ import annotations

from .base import ImageSource, SearchResult, http_get_json

API_URL = "https://api.openverse.org/v1/images/"


class OpenverseSource(ImageSource):
    name = "openverse"
    prefix = "ov"
    search_sleep = 3.0  # 匿名アクセスのレート制限対策

    def search(self, query: str, page: int) -> list[SearchResult]:
        data = http_get_json(
            API_URL,
            params={
                "q": query,
                "page_size": 20,
                "page": page,
                "category": "photograph",  # イラスト・デジタルアートを除外
            },
        )
        results = []
        for r in data.get("results", []):
            if not r.get("url"):
                continue
            results.append(
                SearchResult(
                    source=self.name,
                    source_id=r["id"],
                    image_url=r["url"],
                    thumbnail_url=r.get("thumbnail") or r["url"],
                    foreign_landing_url=r.get("foreign_landing_url") or "",
                    license=r.get("license") or "",
                    license_version=r.get("license_version") or "",
                    creator=r.get("creator") or "",
                    title=r.get("title") or "",
                    tags=[t.get("name") or "" for t in (r.get("tags") or [])],
                )
            )
        return results
