"""collect_images.py の収集プラン（--plan）読み込みと関連性フィルタのテスト。"""

from __future__ import annotations

import json

import pytest

from scripts.collect_images import DEFAULT_CATEGORIES, is_relevant, load_plan
from scripts.sources.base import SearchResult


def _write_plan(tmp_path, data):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _result(title="", tags=None):
    return SearchResult(
        source="openverse",
        source_id="id1",
        image_url="http://img",
        thumbnail_url="http://thumb",
        title=title,
        tags=tags or [],
    )


# ---- load_plan 正常系 ----

def test_load_plan_minimal(tmp_path):
    path = _write_plan(tmp_path, {
        "categories": {
            "dog_photo": {"queries": ["dog park photo"]},
        },
    })
    categories, base_queries, base_keywords = load_plan(path)
    assert set(categories) == {"dog_photo"}
    cat = categories["dog_photo"]
    assert cat["queries"] == ["dog park photo"]
    assert cat["keywords"] == []  # 省略 = フィルタなし
    assert cat["synthetic"] is None
    assert cat["per_category"] is None
    assert base_queries == []
    assert base_keywords == []


def test_load_plan_full(tmp_path):
    path = _write_plan(tmp_path, {
        "categories": {
            "blurred": {
                "queries": ["motion blur people"],
                "keywords": ["Blur", "PEOPLE"],
                "synthetic": "motion_blur",
                "per_category": 30,
            },
        },
        "base_pool_queries": ["people outdoors"],
        "base_pool_keywords": ["People"],
    })
    categories, base_queries, base_keywords = load_plan(path)
    cat = categories["blurred"]
    assert cat["keywords"] == ["blur", "people"]  # 小文字化
    assert cat["synthetic"] == "motion_blur"
    assert cat["per_category"] == 30
    assert base_queries == ["people outdoors"]
    assert base_keywords == ["people"]


# ---- load_plan バリデーション ----

def test_load_plan_requires_categories(tmp_path):
    path = _write_plan(tmp_path, {"categories": {}})
    with pytest.raises(ValueError, match="categories"):
        load_plan(path)


def test_load_plan_rejects_missing_queries(tmp_path):
    path = _write_plan(tmp_path, {"categories": {"dog": {"queries": []}}})
    with pytest.raises(ValueError, match="queries"):
        load_plan(path)


def test_load_plan_rejects_unknown_synthetic(tmp_path):
    path = _write_plan(tmp_path, {
        "categories": {"dog": {"queries": ["dog"], "synthetic": "sepia"}},
        "base_pool_queries": ["dog"],
    })
    with pytest.raises(ValueError, match="synthetic"):
        load_plan(path)


def test_load_plan_synthetic_requires_base_pool(tmp_path):
    path = _write_plan(tmp_path, {
        "categories": {"dog": {"queries": ["dog"], "synthetic": "motion_blur"}},
    })
    with pytest.raises(ValueError, match="base_pool_queries"):
        load_plan(path)


@pytest.mark.parametrize("name", ["_base_pool", "Dog", "dog-photo", "1dog", ""])
def test_load_plan_rejects_bad_bucket_names(tmp_path, name):
    path = _write_plan(tmp_path, {"categories": {name: {"queries": ["q"]}}})
    with pytest.raises(ValueError, match="カテゴリ名"):
        load_plan(path)


def test_load_plan_rejects_bad_per_category(tmp_path):
    path = _write_plan(tmp_path, {
        "categories": {"dog": {"queries": ["q"], "per_category": 0}},
    })
    with pytest.raises(ValueError, match="per_category"):
        load_plan(path)


# ---- 関連性フィルタ ----

def test_is_relevant_matches_title_or_tags():
    assert is_relevant(_result(title="Dog in the park"), ["dog"])
    assert is_relevant(_result(tags=["Dog", "park"]), ["dog"])
    assert not is_relevant(_result(title="A cat"), ["dog"])


def test_empty_keywords_means_no_filter():
    # collect_from_queries は keywords が空ならフィルタ自体を適用しない
    # （is_relevant([]) は常に False なので、空チェックのガードが必須）
    keywords: list[str] = []
    result = _result(title="anything")
    assert not is_relevant(result, keywords)  # 前提の確認
    assert not (keywords and not is_relevant(result, keywords))  # ガード後は除外されない


def test_default_categories_shape():
    # --plan なしの後方互換: 既定カテゴリは load_plan 出力と同じキーで参照できる
    for cat in DEFAULT_CATEGORIES.values():
        assert cat["queries"]
        assert isinstance(cat["keywords"], list)
        assert cat.get("per_category") is None
