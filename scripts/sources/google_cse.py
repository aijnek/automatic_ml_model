"""Google Custom Search JSON API（画像検索）。要 APIキー + 検索エンジンID。

注意: ライセンス情報が取得できないため license="unknown" で記録される。
無料枠は 100クエリ/日。既定ソースには含めず、--sources google で明示指定
したときのみ使う最終手段。
"""

from __future__ import annotations

import hashlib
import os

from .base import ImageSource, SearchResult, http_get_json

API_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleCseSource(ImageSource):
    name = "google"
    prefix = "gc"
    search_sleep = 2.0
    env_vars = ("GOOGLE_API_KEY", "GOOGLE_CSE_ID")
    in_default_set = False  # ライセンス不明のため明示指定時のみ

    _warned = False

    def search(self, query: str, page: int) -> list[SearchResult]:
        if not GoogleCseSource._warned:
            print(
                "  [google] 注意: ライセンス情報は取得できず license=unknown で"
                "記録されます（無料枠 100クエリ/日）"
            )
            GoogleCseSource._warned = True
        data = http_get_json(
            API_URL,
            params={
                "key": os.environ["GOOGLE_API_KEY"],
                "cx": os.environ["GOOGLE_CSE_ID"],
                "q": query,
                "searchType": "image",
                "safe": "active",
                "num": 10,
                "start": 1 + 10 * (page - 1),
            },
        )
        results = []
        for item in data.get("items", []):
            link = item.get("link")
            if not link:
                continue
            image = item.get("image") or {}
            results.append(
                SearchResult(
                    source=self.name,
                    # 安定したIDがないため画像URLのハッシュで代用
                    source_id=hashlib.md5(link.encode()).hexdigest()[:16],
                    image_url=link,
                    thumbnail_url=image.get("thumbnailLink") or link,
                    foreign_landing_url=image.get("contextLink") or "",
                    license="unknown",
                    title=item.get("title") or "",
                )
            )
        return results
