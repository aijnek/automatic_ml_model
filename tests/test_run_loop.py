from __future__ import annotations

import json

import pytest

from pipeline import run_loop


def _feature(name, type_="scale_1_5", action="new", **kw):
    f = {
        "name": name,
        "type": type_,
        "prompt": f"Question about {name}?",
        "action": action,
        "rationale": "test",
    }
    if type_ == "categorical":
        f.setdefault("choices", kw.pop("choices", ["red", "green", "blue", "other"]))
    f.update(kw)
    return f


UNINFORMATIVE = [
    _feature("brightness"),
    _feature("has_multiple_shapes", "binary"),
    _feature("noise_score"),
]
INFORMATIVE = _feature("dominant_color", "categorical")


class FakeDesigner:
    """1回目は uninformative、2回目以降は dominant_color を追加提案する designer。"""

    def __init__(self, informative_from_call=2):
        self.calls: list[str] = []
        self.informative_from_call = informative_from_call

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if len(self.calls) >= self.informative_from_call:
            feats = [dict(INFORMATIVE)] + [
                {**f, "action": "kept"} for f in UNINFORMATIVE
            ]
        else:
            feats = [dict(f) for f in UNINFORMATIVE]
        return json.dumps({"features": feats})


def make_fake_vlm(label_map: dict[str, str]):
    """dominant_color だけ正解ラベルを返し、他はファイル名ハッシュで決まるノイズ。"""

    def vlm(model, prompt, image_path, format_schema):
        name = image_path.name
        out = {}
        for key in format_schema["properties"]:
            if key == "dominant_color":
                out[key] = label_map[name]
            elif key == "has_multiple_shapes":
                out[key] = hash(name) % 2 == 0
            else:
                out[key] = hash((key, name)) % 5 + 1
        return json.dumps(out)

    return vlm


@pytest.fixture
def env(tmp_path, clf_config, clf_annotations):
    """splits・画像・パス一式を tmp_path に用意する。"""
    dirs = {
        "splits_dir": tmp_path / "splits",
        "images_dir": tmp_path / "images",
        "schemas_dir": tmp_path / "schemas",
        "features_dir": tmp_path / "features",
        "models_dir": tmp_path / "models",
        "results_dir": tmp_path / "results",
        "state_path": tmp_path / "state.json",
    }
    dirs["splits_dir"].mkdir()
    dirs["images_dir"].mkdir()
    clf_annotations.iloc[:70].to_csv(dirs["splits_dir"] / "train.csv", index=False)
    clf_annotations.iloc[70:85].to_csv(dirs["splits_dir"] / "val.csv", index=False)
    clf_annotations.iloc[85:].to_csv(dirs["splits_dir"] / "test.csv", index=False)
    for name in clf_annotations["filename"]:
        (dirs["images_dir"] / name).write_bytes(b"fake")
    label_map = dict(zip(clf_annotations["filename"], clf_annotations["label"]))
    return clf_config, dirs, label_map


def test_pass_on_first_iteration(env):
    cfg, dirs, label_map = env
    designer = FakeDesigner(informative_from_call=1)
    state = run_loop.run(
        cfg, designer_llm_fn=designer, vlm_fn=make_fake_vlm(label_map), **dirs
    )
    assert state["finished"] is True
    assert state["final_iteration"] == 1
    assert state["test_score"] > 0.9
    assert len(designer.calls) == 1
    assert (dirs["results_dir"] / "final_report.md").exists()
    assert (dirs["results_dir"] / "iter1" / "report.md").exists()
    assert (dirs["models_dir"] / "model_v1.txt").exists()
    final_md = (dirs["results_dir"] / "final_report.md").read_text()
    assert "test macro_f1" in final_md
    assert "dominant_color" in final_md


def test_improves_over_iterations(env):
    cfg, dirs, label_map = env
    designer = FakeDesigner(informative_from_call=2)
    state = run_loop.run(
        cfg, designer_llm_fn=designer, vlm_fn=make_fake_vlm(label_map), **dirs
    )
    assert state["finished"] is True
    assert state["final_iteration"] == 2
    assert len(state["history"]) == 2
    assert state["history"][0]["val_score"] < cfg.threshold
    assert state["history"][1]["val_score"] >= cfg.threshold
    assert state["best"]["iteration"] == 2
    # 2回目の designer プロンプトには前回の診断レポートが含まれる
    assert "診断レポート" in designer.calls[1]
    assert "brightness" in designer.calls[1]
    # スキーマが2バージョン保存されている
    assert (dirs["schemas_dir"] / "metadata_v1.json").exists()
    assert (dirs["schemas_dir"] / "metadata_v2.json").exists()


def test_resume_after_interruption(env):
    cfg, dirs, label_map = env
    vlm = make_fake_vlm(label_map)

    # 1回目: iter1（不合格）で停止（中断相当）
    designer1 = FakeDesigner(informative_from_call=999)  # ずっと不合格
    state = run_loop.run(
        cfg, designer_llm_fn=designer1, vlm_fn=vlm, max_iterations=1, **dirs
    )
    assert state["finished"] is False
    assert state["iteration"] == 2
    schema_v1 = (dirs["schemas_dir"] / "metadata_v1.json").read_text()

    # 2回目: 再実行。iter1 はやり直さず iter2 から継続し、今回は合格する
    designer2 = FakeDesigner(informative_from_call=1)
    state = run_loop.run(cfg, designer_llm_fn=designer2, vlm_fn=vlm, **dirs)
    assert state["finished"] is True
    assert state["final_iteration"] == 2
    # designer は iter2 の1回だけ呼ばれ、v1 スキーマは再設計されていない
    assert len(designer2.calls) == 1
    assert (dirs["schemas_dir"] / "metadata_v1.json").read_text() == schema_v1
    # 履歴には iter1（中断前）と iter2 の両方が残る
    assert [h["iteration"] for h in state["history"]] == [1, 2]


def test_finished_state_is_idempotent(env):
    cfg, dirs, label_map = env
    designer = FakeDesigner(informative_from_call=1)
    vlm = make_fake_vlm(label_map)
    state1 = run_loop.run(cfg, designer_llm_fn=designer, vlm_fn=vlm, **dirs)
    # 完了後の再実行は何もせずそのまま返す（designer 追加呼び出しなし）
    state2 = run_loop.run(cfg, designer_llm_fn=designer, vlm_fn=vlm, **dirs)
    assert state2["finished"] is True
    assert len(designer.calls) == 1
    assert state2["test_score"] == state1["test_score"]


def test_missing_splits_raises(tmp_path, clf_config):
    with pytest.raises(FileNotFoundError):
        run_loop.run(
            clf_config,
            splits_dir=tmp_path / "nope",
            images_dir=tmp_path / "images",
            schemas_dir=tmp_path / "schemas",
            features_dir=tmp_path / "features",
            models_dir=tmp_path / "models",
            results_dir=tmp_path / "results",
            state_path=tmp_path / "state.json",
        )
