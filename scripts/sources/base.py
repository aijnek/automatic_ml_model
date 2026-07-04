"""画像検索ソースの共通インターフェース。

各ソースは ImageSource を継承して search(query, page) を実装し、
正規化された SearchResult のリストを返す。出典・ライセンス情報は
data/collection_metadata.csv にそのまま記録される。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import requests

USER_AGENT = "automatic-ml-model-collector/0.1 (ML research dataset collection)"


@dataclass
class SearchResult:
    source: str  # ソース名（"openverse" など）
    source_id: str  # ソース側のID（文字列に正規化）
    image_url: str  # フル解像度ダウンロード用URL
    thumbnail_url: str  # 一覧表示用サムネイルURL
    foreign_landing_url: str = ""  # 作者の掲載元ページ（出典表示・重複排除用）
    license: str = ""  # 例: "by-nc-nd", "CC BY-SA 4.0", "pixabay", "unknown"
    license_version: str = ""
    creator: str = ""
    title: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"


def safe_id(s: str) -> str:
    """source_id をファイル名に使える8文字以内の英数字にする。"""
    return re.sub(r"[^A-Za-z0-9]", "", s)[:8] or "x"


def http_get_json(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """JSON APIをGETする。429は60秒待ってリトライ。"""
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT} | (headers or {}),
        timeout=30,
    )
    if resp.status_code == 429:
        print("    レート制限。60秒待機...")
        time.sleep(60)
        return http_get_json(url, params, headers)
    resp.raise_for_status()
    return resp.json()


class ImageSource:
    name: str = ""  # レジストリキー・CSVの source 列の値
    prefix: str = ""  # ファイル名に入るソース略号（ov/px/pe/wm/fl/gc）
    search_sleep: float = 1.0  # 検索リクエスト間の最小間隔（秒）
    env_vars: tuple[str, ...] = ()  # 必要な環境変数。() はキー不要
    in_default_set: bool = True  # --sources auto に含めるか

    def __init__(self) -> None:
        self._last_call = 0.0

    def available(self) -> tuple[bool, str]:
        """(使用可否, 使えない理由)。APIキー未設定を検出する。"""
        missing = [v for v in self.env_vars if not os.environ.get(v)]
        if missing:
            return False, f"環境変数 {', '.join(missing)} 未設定"
        return True, ""

    def throttle(self) -> None:
        """前回の検索から search_sleep 秒経つまで待つ（ソースごとに独立）。"""
        wait = self._last_call + self.search_sleep - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def search(self, query: str, page: int) -> list[SearchResult]:
        raise NotImplementedError
