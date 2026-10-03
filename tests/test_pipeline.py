"""Tests: scoring rules, dataset sanity, and an end-to-end run on a fake Ollama server.  Run: pytest -q"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import mock_ollama  # noqa: E402

srv, url = mock_ollama.start()
os.environ["OLLAMA_HOST"] = url

import run  # noqa: E402
import score  # noqa: E402


def test_dataset_is_well_formed():
    qs = [json.loads(x) for x in (ROOT / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({q["id"] for q in qs}) == len(qs)
    assert all(q["answers"] for q in qs if q["type"] == "factual")
    assert all(q["note"] for q in qs if q["type"] == "false_premise")


def test_factual_matching():
    assert score.factual_correct("대한민국의 수도는 서울입니다.", ["서울"])
    assert score.factual_correct("HBM은 High Bandwidth Memory의 약자입니다.", ["high bandwidth memory"])
    assert not score.factual_correct("부산입니다.", ["서울"])


def test_keyword_rule():
    assert score.keyword_label("그런 제품은 존재하지 않습니다.") == "rejected"
    assert score.keyword_label("조선의 제30대 왕은 이영입니다.") == "accepted"


def test_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "questions.jsonl").write_text((ROOT / "data" / "questions.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    for mod in (run, score):
        monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(score, "R", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["run", "--models", "mock"]); run.main()
    monkeypatch.setattr(sys, "argv", ["score", "--judge", "mock"]); score.main()
    s = json.loads((tmp_path / "results" / "summary.json").read_text(encoding="utf-8"))
    assert len(s["leaderboard"]) == 2 and (tmp_path / "results" / "leaderboard.png").exists()
