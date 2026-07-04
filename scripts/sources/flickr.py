"""Flickr API (https://www.flickr.com/services/api/)。要無料APIキー。

CC・パブリックドメイン系ライセンスのみを検索する。ライセンスIDは
既存CSVの表記（Openverse形式: "by-nc-nd" 等）に合わせて変換する。
"""

from __future__ import annotations

import os

from .base import ImageSource, SearchResult, http_get_json

API_URL = "https://api.flickr.com/services/rest/"

# ライセンスID -> (license, license_version)
FLICKR_LICENSES = {
    "1": ("by-nc-sa", "2.0"),
    "2": ("by-nc", "2.0"),
    "3": ("by-nc-nd", "2.0"),
    "4": ("by", "2.0"),
    "5": ("by-sa", "2.0"),
    "6": ("by-nd", "2.0"),
    "9": ("cc0", "1.0"),
    "10": ("pdm", "1.0"),
}


class FlickrSource(ImageSource):
    name = "flickr"
    prefix = "fl"
    search_sleep = 1.0  # 上限 3600リクエスト/時
    env_vars = ("FLICKR_API_KEY",)

    def search(self, query: str, page: int) -> list[SearchResult]:
        data = http_get_json(
            API_URL,
            params={
                "method": "flickr.photos.search",
                "api_key": os.environ["FLICKR_API_KEY"],
                "text": query,
                "license": ",".join(FLICKR_LICENSES),
                "content_type": 1,  # 写真のみ
                "media": "photos",
                "safe_search": 1,
                "sort": "relevance",
                "extras": "url_l,url_c,url_m,license,owner_name,tags",
                "per_page": 20,
                "page": page,
                "format": "json",
                "nojsoncallback": 1,
            },
        )
        results = []
        for p in (data.get("photos") or {}).get("photo", []):
            image_url = p.get("url_l") or p.get("url_c") or p.get("url_m")
            if not image_url:
                continue
            license_, version = FLICKR_LICENSES.get(str(p.get("license")), ("", ""))
            results.append(
                SearchResult(
                    source=self.name,
                    source_id=str(p["id"]),
                    image_url=image_url,
                    thumbnail_url=p.get("url_m") or p.get("url_c") or image_url,
                    foreign_landing_url=f"https://www.flickr.com/photos/{p['owner']}/{p['id']}",
                    license=license_,
                    license_version=version,
                    creator=p.get("ownername") or "",
                    title=p.get("title") or "",
                    tags=(p.get("tags") or "").split(),
                )
            )
        return results
