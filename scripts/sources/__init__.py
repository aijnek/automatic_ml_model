"""画像収集ソースのレジストリと APIキー（.env）の読み込み。"""

from __future__ import annotations

import os
from pathlib import Path

from .base import ImageSource, SearchResult, safe_id
from .flickr import FlickrSource
from .google_cse import GoogleCseSource
from .openverse import OpenverseSource
from .pexels import PexelsSource
from .pixabay import PixabaySource
from .wikimedia import WikimediaSource

__all__ = ["ImageSource", "SearchResult", "safe_id", "SOURCES", "get_sources", "load_dotenv"]

SOURCES: dict[str, ImageSource] = {
    s.name: s
    for s in [
        OpenverseSource(),
        PixabaySource(),
        PexelsSource(),
        WikimediaSource(),
        FlickrSource(),
        GoogleCseSource(),
    ]
}


def load_dotenv() -> None:
    """プロジェクトルートの .env を os.environ に読み込む（既存値は上書きしない）。"""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_sources(names: list[str] | None = None) -> list[ImageSource]:
    """使用するソースを解決する。

    names=None は「キー設定済みの既定ソース全部」（google は含まない）。
    明示指定でキー未設定のものは表示してスキップする。未知名は ValueError。
    """
    if names is None:
        candidates = [s for s in SOURCES.values() if s.in_default_set]
    else:
        unknown = set(names) - set(SOURCES)
        if unknown:
            raise ValueError(
                f"未知のソース: {', '.join(sorted(unknown))}（利用可能: {', '.join(SOURCES)}）"
            )
        candidates = [SOURCES[n] for n in names]
    picked = []
    for s in candidates:
        ok, reason = s.available()
        if ok:
            picked.append(s)
        else:
            print(f"[skip] {s.name}: {reason}")
    return picked
