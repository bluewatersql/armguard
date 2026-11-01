import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / p) for p in ("src", "src/3_ml", "src/4_genai")]
import json
import numpy as np, pandas as pd
import common as cm

IL, D = pd.Timestamp("2025-08-22"), pd.Timedelta("1D")


def runs(*spans):
    return pd.DataFrame([(a, IL + s * D, IL + e * D) for a, s, e in spans], columns=["athlete_id", "run_start", "run_end"])


def test_detect_counts_only_days_inside_the_window():
    import holdout as ho
    cases = {"a": (-23, -21), "b": (-22, -20), "c": (-2, -1), "d": (-1, 0), "e": (-40, -10), "f": (-5, -5)}
    il = pd.DataFrame([(k, k, IL) for k in cases], columns=["key", "athlete_id", "il_date"])
    d = ho.detect(runs(*[(k, s, e) for k, (s, e) in cases.items()]), il)
    assert d.det.tolist() == [False, True, True, False, True, False]  # a: only IL-21 inside; d: only IL-1 inside
    assert d.lead.tolist()[1:3] == [21, 2] and d.lead.iloc[4] == 21 and np.isnan(d.lead.iloc[0])


def test_hard_negative_window_edges():
    import holdout as ho
    sp = {"a": (-4, -3), "b": (-3, -2), "c": (9, 10), "d": (10, 11)}
    hn = ho.windows(pd.DataFrame([(k, k, IL) for k in sp], columns=["key", "athlete_id", "il_date"]), -3, 10)
    assert ho.first_hit(runs(*[(k, s, e) for k, (s, e) in sp.items()]), hn).notna().tolist() == [False, True, True, False]
    assert ho.first_hit(runs(), hn).isna().all()


def test_evaluate_lead_hard_negatives_null_and_a4():
    import holdout as ho
    hot = {"X": (-10, -9), "Y": (-5,), "Z": (-30, -29, -25), "H": (9,), "K": (1, 2), "N": (-3, -2), "M": (-3, -2)}
    rows = pd.DataFrame([(a, IL + d * D, .5 if d in h or a == "Z" and -30 <= d <= -15 else 0., -2. if d in h else 0.)
                         for a, h in hot.items() for d in range(-40, 12)], columns=["athlete_id", "feature_date", "risk_score", "velo_delta"])
    rows = rows[(rows.athlete_id != "M") | (rows.feature_date >= IL - 3 * D)]  # M: 3 eligible days in the window
    K = {"X": "i", "Y": "i", "Z": "i", "H": "cold_weather", "K": "dead_arm_note"}
    inc = pd.DataFrame([(a, a, "injury" if s == "i" else "hard_negative", s, IL) for a, s in K.items()],
                       columns=["key", "athlete_id", "kind", "sub_kind", "il_date"])
    per, neg, S = ho.evaluate(rows, inc, .13, {"K": False}, never={"N"})
    assert per.m_det.tolist() == [1, 0, 1] and per.m_lead[[0, 2]].tolist() == [10, 21] and per.r_det.tolist() == [1, 0, 0]
    assert S["m"]["det"] == 2 and S["m"]["lead"] == 15.5 and S["m"]["pairs"] == 2 and S["m"]["null"] == 1
    assert S["m"]["strict_pairs"] == 1 and S["m"]["strict_athletes"] == 1
    assert neg.m_run.tolist() == [0, 1] and neg.m_fa.tolist() == [0, 0] and neg.r_fa.tolist() == [0, 1]
    assert [ok for _, ok, _ in ho.a4(S, 3, True)] == [False, True, True, None, True, True]
    e = ho.evaluate(rows.assign(risk_score=0., velo_delta=0.), inc, .13, {}, never=set())[2]["m"]
    assert e["det"] == e["null"] == 0 and np.isnan(e["lead"])
    one = rows[(rows.athlete_id != "M") | (rows.feature_date >= IL - D)]  # M with 1 eligible day: pair dropped
    assert ho.evaluate(one, inc, .13, {"K": False}, never=set())[2]["m"]["pairs"] == 1


def test_config_hash_ignores_model_version_and_tracks_config():
    import holdout as ho
    base = dict(features=["a", "b"], params={"max_depth": 3}, high=.13, elevated=.078, seed=4242, version="2.0", labels=("v1", "{}"))
    h = ho.config_hash(**base)
    assert h == ho.config_hash(**{**base, "high": np.float64(.13), "seed": np.int64(4242)}) and len(h) == 16
    assert all(h != ho.config_hash(**{**base, k: v}) for k, v in
               (("features", ["a"]), ("params", {"max_depth": 4}), ("high", .17), ("seed", 1), ("version", "2.1"),
                ("labels", ("v2", "{}")), ("labels", ("v1", "{ }"))))


def test_genai_leak_window_and_fallback_fails_a5():
    import genai_eval as g
    n = ["Sharp pain on release, numbness in two fingers."]
    assert g.leaks("Felt sharp PAIN on release, numbness in his arm", n) and not g.leaks("pain on release, numbness noted", n)
    rec = lambda fb: {"inputs": {"facts": json.dumps({"velo_delta": -1.8}), "athlete_id": "A"},
                      "outputs": {"response": "Down 1.8 mph"}, "expectations": {"action": "monitor", "fallback": fb}}
    assert g.checks(cm, rec(False), {"A": n}) == {"brief_grounded": True, "no_medical_leak": True, "action_in_playbook": True}
    assert g.checks(cm, rec(True), {"A": n}) == {"brief_grounded": False, "no_medical_leak": True, "action_in_playbook": False}


def test_open_cases_refreshes_open_cases_only_and_counts_inserts():
    sys.path.insert(0, str(R / "src/7_app"))
    import cases as cs
    from sqlalchemy import create_engine
    with create_engine("sqlite://").begin() as cx:
        for s in cs.ddl(None):
            cx.exec_driver_sql(s)
        D, rows = "2025-08-15", [{"athlete_id": "A22", "priority": "P2", "risk_score": .08}, {"athlete_id": "A05", "priority": "P2", "risk_score": .3}]
        assert cs.open_cases(cx, rows, D) == 2
        b = cs.find(cx, "A05", D)
        cs.transition(cx, b["case_id"], "acknowledged", "t", b["updated_at"])
        a0, b0 = cs.find(cx, "A22", D), cs.find(cx, "A05", D)
        assert cs.open_cases(cx, [{**r, "priority": "P1", "risk_score": .9} for r in rows], D) == 0  # re-run opens 0
        a, b = cs.find(cx, "A22", D), cs.find(cx, "A05", D)
        assert (a["priority"], a["risk_score"], a["status"], a["updated_at"]) == ("P1", .9, "open", a0["updated_at"])
        assert b == b0  # non-open case untouched
        assert cs.open_cases(cx, rows + [{"athlete_id": "A07", "priority": "P3", "risk_score": None}], D) == 1
        assert [r["priority"] for r in cs.queue(cx)] == ["P2", "P2", "P3"]
