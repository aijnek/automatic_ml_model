from __future__ import annotations

import json

import pytest

from pipeline import designer


def _feature(name, type_="scale_1_5", action="new", **kw):
    f = {
        "name": name,
        "type": type_,
        "prompt": f"Rate {name} from 1 to 5.",
        "action": action,
        "rationale": "test",
    }
    if type_ == "categorical":
        f["choices"] = kw.pop("choices", ["a", "b"])
    f.update(kw)
    return f


def _valid_proposal(n=5):
    return {"features": [_feature(f"feat_{i}") for i in range(n)]}


# ---- extract_json ----

def test_extract_json_plain():
    obj = designer.extract_json(json.dumps(_valid_proposal()))
    assert len(obj["features"]) == 5


def test_extract_json_code_fence():
    text = f"以下が結果です。\n```json\n{json.dumps(_valid_proposal())}\n```\n以上。"
    assert len(designer.extract_json(text)["features"]) == 5


def test_extract_json_surrounded_by_prose():
    text = f"Here is the schema: {json.dumps(_valid_proposal())} Hope it helps!"
    assert len(designer.extract_json(text)["features"]) == 5


def test_extract_json_braces_inside_strings():
    prop = _valid_proposal()
    prop["features"][0]["prompt"] = 'Answer in JSON like {"x": 1}. Rate it.'
    assert designer.extract_json(json.dumps(prop))["features"][0]["prompt"].startswith("Answer")


def test_extract_json_no_json():
    with pytest.raises(designer.SchemaValidationError):
        designer.extract_json("すみません、わかりません。")


# ---- validate_schema ----

def test_validate_rejects_bad_name():
    prop = _valid_proposal()
    prop["features"][0]["name"] = "BadName"
    with pytest.raises(designer.SchemaValidationError):
        designer.validate_schema(prop)


def test_validate_rejects_duplicate_names():
    prop = {"features": [_feature("dup"), _feature("dup")] + _valid_proposal()["features"]}
    with pytest.raises(designer.SchemaValidationError):
        designer.validate_schema(prop)


def test_validate_rejects_bad_type():
    prop = _valid_proposal()
    prop["features"][0]["type"] = "integer"
    with pytest.raises(designer.SchemaValidationError):
        designer.validate_schema(prop)


def test_validate_rejects_categorical_without_choices():
    prop = _valid_proposal()
    prop["features"][0] = _feature("cat_feat", "categorical")
    del prop["features"][0]["choices"]
    with pytest.raises(designer.SchemaValidationError):
        designer.validate_schema(prop)


def test_validate_rejects_too_few_active():
    prop = {"features": [_feature("a"), _feature("b"), _feature("c", action="removed")]}
    with pytest.raises(designer.SchemaValidationError):
        designer.validate_schema(prop)


def test_validate_accepts_sample_schema(sample_schema):
    designer.validate_schema(sample_schema)


# ---- merge_schema ----

def test_merge_kept_uses_previous_definition(sample_schema):
    proposed = {
        "features": [
            # kept だがプロンプトが揺れている → 前回定義を使うべき
            _feature("brightness", action="kept", prompt="DIFFERENT PROMPT"),
            _feature("dominant_color", "categorical", action="kept",
                     choices=["red", "green", "blue", "other"]),
            _feature("has_multiple_shapes", "binary", action="kept"),
            _feature("shape_area_ratio", "float", action="kept"),
            _feature("new_feat", action="new"),
        ]
    }
    merged = designer.merge_schema(sample_schema, proposed, version=2)
    by_name = {f["name"]: f for f in merged["features"]}
    prev_prompt = next(
        f["prompt"] for f in sample_schema["features"] if f["name"] == "brightness"
    )
    assert by_name["brightness"]["prompt"] == prev_prompt
    assert by_name["new_feat"]["action"] == "new"
    assert merged["version"] == 2


def test_merge_removed_and_unmentioned(sample_schema):
    proposed = {
        "features": [
            _feature("brightness", action="removed"),
            _feature("edge_sharpness", action="new"),
        ]
    }
    merged = designer.merge_schema(sample_schema, proposed, version=2)
    by_name = {f["name"]: f for f in merged["features"]}
    assert by_name["brightness"]["action"] == "removed"
    assert "brightness" not in {f["name"] for f in designer.active_features(merged)}
    # 提案に現れなかった特徴量は kept 扱いで残る
    assert by_name["dominant_color"]["action"] == "kept"
    assert by_name["edge_sharpness"]["action"] == "new"


def test_merge_modified_overrides(sample_schema):
    proposed = {
        "features": [
            _feature("brightness", action="modified", prompt="New anchored prompt 1-5."),
        ]
    }
    merged = designer.merge_schema(sample_schema, proposed, version=3)
    by_name = {f["name"]: f for f in merged["features"]}
    assert by_name["brightness"]["prompt"] == "New anchored prompt 1-5."
    assert by_name["brightness"]["action"] == "modified"


# ---- design_schema (LLMモック) ----

def test_design_schema_initial(clf_config):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return json.dumps(_valid_proposal())

    schema = designer.design_schema(clf_config, version=1, llm_fn=fake_llm)
    assert schema["version"] == 1
    assert len(schema["features"]) == 5
    assert "分類" in calls[0]
    assert "red" in calls[0]  # クラス名がプロンプトに入る


def test_design_schema_retries_on_invalid_json(clf_config):
    outputs = ["not json at all", json.dumps(_valid_proposal())]
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return outputs[len(calls) - 1]

    schema = designer.design_schema(clf_config, version=1, llm_fn=fake_llm)
    assert len(calls) == 2
    assert "エラーで拒否されました" in calls[1]
    assert len(schema["features"]) == 5


def test_design_schema_gives_up_after_retries(clf_config):
    def fake_llm(prompt):
        return "garbage"

    with pytest.raises(designer.SchemaValidationError):
        designer.design_schema(clf_config, version=1, llm_fn=fake_llm)


def test_design_schema_revision_includes_report(clf_config, sample_schema):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return json.dumps(
            {"features": [_feature("brightness", action="removed"),
                          _feature("extra_a"), _feature("extra_b")]}
        )

    schema = designer.design_schema(
        clf_config,
        version=2,
        prev_schema=sample_schema,
        prev_report_md="val macro-F1: 0.55\n重要度最下位: brightness",
        llm_fn=fake_llm,
    )
    assert "val macro-F1: 0.55" in calls[0]
    assert "dominant_color" in calls[0]  # 現行スキーマもプロンプトに入る
    names = {f["name"] for f in designer.active_features(schema)}
    assert "brightness" not in names
    assert {"extra_a", "extra_b"} <= names


def test_revision_prompt_includes_reliability_policy(clf_config, sample_schema):
    prompt = designer.build_revision_prompt(clf_config, sample_schema, "# レポート")
    assert "抽出信頼性" in prompt
    assert "不安定" in prompt and "縮退" in prompt and "抽出失敗" in prompt


def test_format_spec_requires_objective_prompts(clf_config):
    prompt = designer.build_initial_prompt(clf_config)
    assert "客観的な質問" in prompt
