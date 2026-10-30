import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / p) for p in ("src", "src/1_lakeflow", "src/3_ml", "src/7_app", "tools")]
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


import numpy as np, pandas as pd
import sim_core as sc
import simulate_club as scl

def _game(home=136, temp="72"):
    pe = lambda n, v: {"isPitch": True, "pitchNumber": n, "details": {"type": {"code": "FF"}},
        "pitchData": {"startSpeed": v, "extension": 6.4, "breaks": {"spinRate": 2300}, "coordinates": {"z0": 5.9}}}
    return {"gamePk": 7, "gameData": {"datetime": {"officialDate": "2025-07-01"}, "weather": {"temp": temp},
            "teams": {"home": {"id": home}, "away": {"id": 9}},
            "players": {"ID11": {"id": 11, "fullName": "Real Person", "pitchHand": {"code": "L"}}}},
            "liveData": {"plays": {"allPlays": [
                {"about": {"isTopInning": True}, "matchup": {"pitcher": {"id": 11}}, "playEvents": [pe(1, 95.1), pe(2, 94.8), {"isPitch": False}]},
                {"about": {"isTopInning": False}, "matchup": {"pitcher": {"id": 22}}, "playEvents": [pe(1, 90.0)]}]}}}

def test_extract_club_side_and_hash():
    rows = sc.extract_club_pitches(_game(), 136, 4242)
    assert {r["mlbam_id"] for r in rows} == {11} and len(rows) == 2
    assert len(rows[0]["pitch_uid"]) == 16 and "7-0-1" not in rows[0]["pitch_uid"]

def test_extract_handles_missing_fields():
    g = _game(temp=""); g["liveData"]["plays"]["allPlays"][0]["playEvents"][0]["pitchData"] = {}
    r = sc.extract_club_pitches(g, 136, 4242)[0]
    assert r["release_speed"] is None and r["temp_f"] is None

def _apps(n=30, k=20):
    d = pd.Timestamp("2025-04-01")
    return pd.DataFrame([{"mlbam_id": 1000 + p, "game_date": d + pd.Timedelta(days=5 * j + p % 5),
                          "pitches": 90 if p % 2 else 20, "fb_velo": 95.0, "app_role": "SP" if p % 2 else "RP"}
                         for p in range(n) for j in range(k * 2)])

def test_select_incidents_protocol():
    a, b = sc.select_incidents(_apps(), np.random.default_rng(4242)), sc.select_incidents(_apps(), np.random.default_rng(4242))
    pd.testing.assert_frame_equal(a, b)
    inj, neg = a[a.kind == "injury"], a[a.kind == "hard_negative"]
    assert (inj.split == "train").sum() == 8 and (inj.split == "holdout").sum() == 8
    assert inj[inj.split == "train"].il_date.between("2025-05-15", "2025-06-30").all()
    assert inj[inj.split == "holdout"].il_date.between("2025-07-20", "2025-09-20").all()
    assert (inj.il_date - inj.true_onset_date).dt.days.between(10, 21).all()
    assert sorted(neg[neg.split == "holdout"].sub_kind) == ["cold_weather", "cold_weather", "dead_arm_note", "role_change"]
    assert sorted(neg[neg.split == "train"].sub_kind) == ["cold_weather", "role_change"]
    hero = a[a.is_hero]
    assert len(hero) == 1 and hero.il_date.iloc[0] == pd.Timestamp("2025-08-22")

def test_precursor_concave_ramp_and_suppression():
    p = pd.DataFrame({"mlbam_id": [1] * 3, "release_speed": [95.0] * 3, "spin_rate": [2300.0] * 3, "release_z": [6.0] * 3,
                      "extension": [6.4] * 3, "temp_f": [70] * 3, "pitch_uid": list("abc"),
                      "game_date": pd.to_datetime(["2025-07-01", "2025-07-11", "2025-07-25"])})
    inc = pd.DataFrame([{"incident_id": "i1", "mlbam_id": 1, "kind": "injury", "sub_kind": "", "true_onset_date": pd.Timestamp("2025-07-01"),
                         "il_date": pd.Timestamp("2025-07-21"), "il_days": 15, "velo_drop": 2.0, "spin_drop": 100.0,
                         "rel_z_drop": 0.1, "ext_drop": 0.1}])
    out = sc.apply_precursors(p, inc).set_index("pitch_uid")
    assert out.loc["a", "release_speed"] == 95.0 and abs(out.loc["b", "release_speed"] - (95 - 2 * 0.5 ** 0.5)) < 1e-6
    assert "c" not in out.index

def test_roster_rejects_real_names_and_il_text():
    drawn = scl.make_roster([11], np.random.default_rng(1), {11: "R"}, set()).display_name.iloc[0]
    ros = scl.make_roster([11], np.random.default_rng(1), {11: "R"}, {drawn})
    assert ros.display_name.iloc[0] != drawn and ros.athlete_id.iloc[0] == "A01"
    inc = pd.DataFrame([{"incident_id": "i1", "mlbam_id": 11, "kind": "injury", "il_date": pd.Timestamp("2025-07-21"),
                         "il_days": 15, "body_part": "forearm"}])
    ev = scl.make_il_events(inc, ros)
    assert ev[ev.action == "placed"].description.iloc[0].startswith(f"Cascadia Ospreys placed RHP {ros.display_name.iloc[0]} on the 15-day injured list.")
    assert (ev.action == "activated").sum() == 1

def test_alert_runs_and_incident_recall():
    import train as tr
    def days(a, ds):
        return [{"athlete_id": a, "feature_date": pd.Timestamp("2025-05-01") + pd.Timedelta(days=k), "f": True} for k in ds]
    df = pd.DataFrame(days("A", [3, 4, 5]) + days("B", [10]) + days("C", [0, 1]) + days("D", [40, 41]))
    runs = tr.alert_runs(df, "f")
    assert sorted(runs.athlete_id) == ["A", "C", "D"]
    il = pd.DataFrame({"athlete_id": list("ABCD"), "il_date": pd.Timestamp("2025-05-01") + pd.to_timedelta([15, 15, 40, 45], "D")})
    assert tr.incident_recall(runs, il) == 0.5
