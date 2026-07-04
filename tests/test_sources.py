from __future__ import annotations

import csv

import pytest

from scripts import collect_images
from scripts.sources import SOURCES, get_sources
from scripts.sources import base as sources_base
from scripts.sources import flickr, google_cse, openverse, pexels, pixabay, wikimedia
from scripts.sources.base import SearchResult, safe_id


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _stub_get(monkeypatch, payload):
    """base.http_get_json が使う requests.get を差し替える。"""
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params or {}
        captured["headers"] = headers or {}
        return _FakeResp(payload)

    monkeypatch.setattr(sources_base.requests, "get", fake_get)
    return captured


# ---- SearchResult / safe_id ----

def test_search_result_key():
    r = SearchResult(source="pixabay", source_id="123", image_url="u", thumbnail_url="t")
    assert r.key == "pixabay:123"


def test_safe_id_strips_and_truncates():
    assert safe_id("c417a700-20e2-4e32") == "c417a700"
    assert safe_id("abc") == "abc"
    assert safe_id("!!!") == "x"


# ---- openverse ----

def test_openverse_normalization(monkeypatch):
    _stub_get(monkeypatch, {
        "results": [
            {
                "id": "c417a700-20e2",
                "url": "http://img",
                "thumbnail": "http://thumb",
                "foreign_landing_url": "http://landing",
                "license": "by-nc-nd",
                "license_version": "2.0",
                "creator": "someone",
                "title": "kids playing",
                "tags": [{"name": "children"}, {"name": None}],
            },
            {"id": "no-url", "url": None},
        ]
    })
    results = openverse.OpenverseSource().search("kids", 1)
    assert len(results) == 1
    r = results[0]
    assert r.source == "openverse"
    assert r.source_id == "c417a700-20e2"
    assert r.tags == ["children", ""]
    assert r.license == "by-nc-nd"


# ---- pixabay ----

def test_pixabay_normalization(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "k")
    captured = _stub_get(monkeypatch, {
        "hits": [
            {
                "id": 12345,
                "largeImageURL": "http://large",
                "webformatURL": "http://web",
                "pageURL": "http://page",
                "user": "author",
                "tags": "kids, playground, fun",
            }
        ]
    })
    results = pixabay.PixabaySource().search("kids", 2)
    r = results[0]
    assert r.source_id == "12345"  # 数値IDが文字列化される
    assert r.image_url == "http://large"
    assert r.tags == ["kids", "playground", "fun"]
    assert r.license == "pixabay"
    assert captured["params"]["image_type"] == "photo"
    assert captured["params"]["page"] == 2


# ---- pexels ----

def test_pexels_normalization(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "secret")
    captured = _stub_get(monkeypatch, {
        "photos": [
            {
                "id": 99,
                "src": {"large2x": "http://l2x", "medium": "http://med", "original": "http://orig"},
                "url": "http://page",
                "photographer": "p",
                "alt": "children in park",
            },
            {"id": 100, "src": {"original": "http://orig-only"}},
        ]
    })
    results = pexels.PexelsSource().search("kids", 1)
    assert results[0].image_url == "http://l2x"  # large2x 優先
    assert results[1].image_url == "http://orig-only"  # フォールバック
    assert results[0].license == "pexels"
    assert captured["headers"]["Authorization"] == "secret"


# ---- wikimedia ----

def test_wikimedia_normalization(monkeypatch):
    _stub_get(monkeypatch, {
        "query": {
            "pages": {
                "1": {
                    "pageid": 111,
                    "title": "File:Kids at playground.jpg",
                    "imageinfo": [{
                        "mime": "image/jpeg",
                        "thumburl": "http://thumb1600",
                        "descriptionurl": "http://desc",
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": '<a href="x">Some Artist</a>'},
                            "ImageDescription": {"value": "children playing"},
                        },
                    }],
                },
                "2": {
                    "pageid": 222,
                    "title": "File:Map.tif",
                    "imageinfo": [{"mime": "image/tiff", "thumburl": "http://t"}],
                },
            }
        }
    })
    results = wikimedia.WikimediaSource().search("kids", 1)
    assert len(results) == 1  # TIFF はスキップ
    r = results[0]
    assert r.source_id == "111"
    assert r.image_url == "http://thumb1600"
    assert r.license == "CC BY-SA 4.0"
    assert r.creator == "Some Artist"  # HTMLタグ除去
    assert "Kids at playground.jpg" in r.title and "children playing" in r.title


# ---- flickr ----

def test_flickr_normalization(monkeypatch):
    monkeypatch.setenv("FLICKR_API_KEY", "k")
    _stub_get(monkeypatch, {
        "photos": {
            "photo": [
                {
                    "id": "51804648145",
                    "owner": "55453048@N03",
                    "license": "4",
                    "url_c": "http://c",
                    "url_m": "http://m",
                    "ownername": "harry",
                    "title": "kindergarten",
                    "tags": "kids playground",
                },
                {"id": "2", "owner": "o", "license": "9", "url_m": "http://m2", "tags": ""},
                {"id": "3", "owner": "o", "license": "1", "tags": ""},  # URLなし
            ]
        }
    })
    results = flickr.FlickrSource().search("kids", 1)
    assert len(results) == 2
    assert results[0].image_url == "http://c"  # url_l なし -> url_c フォールバック
    assert (results[0].license, results[0].license_version) == ("by", "2.0")
    assert (results[1].license, results[1].license_version) == ("cc0", "1.0")
    assert results[0].foreign_landing_url == (
        "https://www.flickr.com/photos/55453048@N03/51804648145"
    )
    assert results[0].tags == ["kids", "playground"]


# ---- google ----

def test_google_normalization(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    captured = _stub_get(monkeypatch, {
        "items": [
            {
                "link": "http://img.example/a.jpg",
                "title": "kids",
                "image": {"thumbnailLink": "http://thumb", "contextLink": "http://ctx"},
            }
        ]
    })
    results = google_cse.GoogleCseSource().search("kids", 2)
    r = results[0]
    assert r.license == "unknown"
    assert len(r.source_id) == 16  # URLハッシュ
    assert captured["params"]["searchType"] == "image"
    assert captured["params"]["start"] == 11  # page 2 -> start=11


# ---- available / get_sources ----

def test_available_missing_key(monkeypatch):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    ok, reason = SOURCES["pixabay"].available()
    assert not ok
    assert "PIXABAY_API_KEY" in reason


def test_available_with_key(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "k")
    assert SOURCES["pixabay"].available() == (True, "")


def test_keyless_sources_always_available():
    assert SOURCES["openverse"].available()[0]
    assert SOURCES["wikimedia"].available()[0]


def test_google_requires_both_vars(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    ok, reason = SOURCES["google"].available()
    assert not ok
    assert "GOOGLE_CSE_ID" in reason


def test_get_sources_auto_excludes_google(monkeypatch):
    for var in ("PIXABAY_API_KEY", "PEXELS_API_KEY", "FLICKR_API_KEY"):
        monkeypatch.setenv(var, "k")
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("GOOGLE_CSE_ID", "cx")
    names = [s.name for s in get_sources(None)]
    assert "google" not in names
    assert {"openverse", "pixabay", "pexels", "wikimedia", "flickr"} == set(names)


def test_get_sources_skips_missing_keys(monkeypatch, capsys):
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    names = [s.name for s in get_sources(["openverse", "pixabay"])]
    assert names == ["openverse"]
    assert "[skip] pixabay" in capsys.readouterr().out


def test_get_sources_unknown_name():
    with pytest.raises(ValueError, match="未知のソース"):
        get_sources(["openverse", "nope"])


# ---- CSVスキーマ移行 ----

OLD_HEADER = (
    "filename,category,is_synthetic,openverse_id,base_filename,image_url,"
    "foreign_landing_url,license,license_version,creator,query,augmentation_params"
)


def test_load_metadata_migrates_old_schema(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / "collection_metadata.csv"
    csv_path.write_text(
        OLD_HEADER + "\n"
        "good_0001_c417a700.jpg,good,false,c417a700-20e2,,http://img,http://landing,"
        "by-nc-nd,2.0,someone,kids,\n"
        "backlit_syn_0001.jpg,backlit,true,,good_0001_c417a700.jpg,http://img,"
        "http://landing,by-nc-nd,2.0,someone,,backlit:gamma=1.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(collect_images, "METADATA_CSV", csv_path)

    rows = collect_images.load_metadata()
    assert rows[0]["source"] == "openverse"
    assert rows[0]["source_id"] == "c417a700-20e2"
    assert "openverse_id" not in rows[0]
    assert rows[1]["source"] == ""  # 加工画像はソースなし
    assert rows[1]["source_id"] == ""
    assert (tmp_path / "collection_metadata.csv.bak").exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == collect_images.METADATA_FIELDS

    # 2回目は移行済みなので no-op（バックアップも上書きされない）
    bak_before = (tmp_path / "collection_metadata.csv.bak").read_text(encoding="utf-8")
    capsys.readouterr()
    rows2 = collect_images.load_metadata()
    assert rows2 == rows
    assert "移行" not in capsys.readouterr().out
    assert (tmp_path / "collection_metadata.csv.bak").read_text(encoding="utf-8") == bak_before


def test_load_metadata_new_schema_passthrough(tmp_path, monkeypatch):
    csv_path = tmp_path / "collection_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=collect_images.METADATA_FIELDS)
        writer.writeheader()
        writer.writerow(
            {k: "" for k in collect_images.METADATA_FIELDS}
            | {"filename": "good_0001_px_123.jpg", "source": "pixabay", "source_id": "123"}
        )
    monkeypatch.setattr(collect_images, "METADATA_CSV", csv_path)
    rows = collect_images.load_metadata()
    assert rows[0]["source"] == "pixabay"
    assert not (tmp_path / "collection_metadata.csv.bak").exists()


def test_load_metadata_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_images, "METADATA_CSV", tmp_path / "nope.csv")
    assert collect_images.load_metadata() == []


# ---- is_relevant ----

def test_is_relevant_uses_title_and_tags():
    r = SearchResult(
        source="pixabay", source_id="1", image_url="u", thumbnail_url="t",
        title="Sunny Day", tags=["Kindergarten", "outdoor"],
    )
    assert collect_images.is_relevant(r, ["kindergarten"])
    assert not collect_images.is_relevant(r, ["classroom"])
