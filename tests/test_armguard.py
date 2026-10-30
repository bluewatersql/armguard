import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / p) for p in ("src", "src/1_lakeflow", "src/7_app", "tools")]
import pytest
import common as cm

F = {"display_name": "Dane Corliss", "velo_delta": -1.8, "pitches_7d": 104, "tier": "HIGH", "note_flag": "soreness",
     "risk_score": 0.81, "top_drivers": "velo_delta -1.9 mph; acwr 1.6"}

def test_baseline_requires_history():
    assert cm.baseline_flag(-1.5) and not cm.baseline_flag(-1.4) and not cm.baseline_flag(None)

def test_tier_and_priority():
    assert [cm.tier(s, .6, .36) for s in (.6, .36, .35)] == ["HIGH", "ELEVATED", "NORMAL"]
    assert cm.priority("NORMAL", "red_flag") == "P1" and cm.priority("HIGH", "soreness") == "P1"
    assert cm.priority("HIGH", None) == "P2" and cm.priority("ELEVATED", "fatigue") == "P2"
    assert cm.priority("ELEVATED", "none") == "P3" and cm.priority("NORMAL", "soreness") is None

def test_finalize_brief_fallbacks():
    ok = {"headline": "Velo down", "why": "Down 1.8 mph", "recommended_action": "limit_pitches", "confidence_note": "x"}
    assert cm.finalize_brief(ok, F)["fallback"] is False
    for bad in (None, {}, {**ok, "recommended_action": "shut_down"}):
        out = cm.finalize_brief(bad, F)
        assert out["fallback"] and out["recommended_action"] in cm.PLAYBOOK and "Dane Corliss" in out["headline"]

def test_numbers_grounded_realistic():
    for s in ("P1: velocity down 1.8 mph over 3 outings; 104 pitches in 7 days.",
              "Velo down 1.9 mph; ACWR 1.6.", "Risk score 81%."):
        assert cm.numbers_grounded(s, F), s
    assert not cm.numbers_grounded("Limit to 75 pitches.", F)
    assert not cm.numbers_grounded("Down 3.2 mph.", F)

def test_bootstrap_uses_argv0(monkeypatch, tmp_path):
    script = tmp_path / "task" / "job.py"
    monkeypatch.setattr(sys, "argv", [str(script)])
    monkeypatch.setattr(sys, "path", list(sys.path))
    cm.bootstrap()
    assert sys.path[:2] == [str(script.parent), str(tmp_path)]


import json
import ingest_mlb as im

def test_final_games_only():
    S = {"dates": [{"games": [{"gamePk": 1, "status": {"abstractGameState": "Final"}},
        {"gamePk": 2, "status": {"abstractGameState": "Preview"}},
        {"gamePk": 3, "status": {"abstractGameState": "Final", "detailedState": "Postponed"}},
        {"gamePk": 1, "status": {"abstractGameState": "Final"}}]}]}
    assert im.final_game_pks(S) == [1]

def test_month_windows():
    assert im.month_windows("2025-03-15", "2025-05-02") == [("2025-03-15", "2025-03-31"),
        ("2025-04-01", "2025-04-30"), ("2025-05-01", "2025-05-02")]

def test_skip_existing(tmp_path):
    p = tmp_path / "g.json"; assert im.needs_fetch(str(p))
    p.write_text(json.dumps({"a": 1})); assert not im.needs_fetch(str(p))
    (tmp_path / "e.json").write_text(""); assert im.needs_fetch(str(tmp_path / "e.json"))
