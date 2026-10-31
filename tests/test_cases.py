import sys, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "src/7_app"), str(R / "src/6_genie"), str(R / "src/3_ml")]
import pytest
from sqlalchemy import create_engine
import cases as cs
import numpy as np, pandas as pd
import train

@pytest.fixture
def conn():
    eng = create_engine("sqlite://")
    with eng.begin() as cx:
        for s in cs.ddl(None): cx.exec_driver_sql(s)
    with eng.begin() as cx:
        yield cx

ROWS = [{"athlete_id": "A01", "priority": "P1", "risk_score": .81}, {"athlete_id": "A02", "priority": "P3", "risk_score": .35}]

def test_open_cases_idempotent(conn):
    assert cs.open_cases(conn, ROWS, "2025-08-15") == 2 and cs.open_cases(conn, ROWS, "2025-08-15") == 0
    assert [r["priority"] for r in cs.queue(conn)] == ["P1", "P3"]

def test_transition_and_events(conn):
    cs.open_cases(conn, ROWS, "2025-08-15"); c = cs.queue(conn)[0]
    out = cs.transition(conn, c["case_id"], "action_set", "coach", c["updated_at"], action="limit_pitches", pitch_limit=75)
    assert out["status"] == "action_set" and out["pitch_limit"] == 75
    assert cs.events(conn, c["case_id"])[-1]["to_status"] == "action_set"

def test_stale_update_rejected(conn):
    cs.open_cases(conn, ROWS, "2025-08-15"); c = cs.queue(conn)[0]
    cs.transition(conn, c["case_id"], "acknowledged", "trainer", c["updated_at"])
    with pytest.raises(cs.StaleUpdate):
        cs.transition(conn, c["case_id"], "cleared", "coach", c["updated_at"])

def test_illegal_transition_rejected_and_reset(conn):
    cs.open_cases(conn, ROWS, "2025-08-15"); c = cs.queue(conn)[0]
    done = cs.transition(conn, c["case_id"], "cleared", "dr", c["updated_at"])
    with pytest.raises(cs.IllegalTransition):
        cs.transition(conn, c["case_id"], "acknowledged", "coach", done["updated_at"])
    assert cs.reset_cases(conn, "2025-08-15") == 1 and cs.queue(conn)[0]["status"] == "open"

def test_stale_checked_before_legality(conn):
    cs.open_cases(conn, ROWS, "2025-08-15"); c = cs.queue(conn)[0]
    cs.transition(conn, c["case_id"], "cleared", "dr", c["updated_at"])
    with pytest.raises(cs.StaleUpdate):  # stale AND illegal (cleared -> acknowledged): stale wins
        cs.transition(conn, c["case_id"], "acknowledged", "coach", c["updated_at"])

def test_queue_hides_smoke_and_filters_date(conn):
    cs.open_cases(conn, ROWS, "2025-08-15"); cs.open_cases(conn, [ROWS[0]], "2025-08-16")
    cs.open_cases(conn, [{"athlete_id": cs.SMOKE, "priority": "P1", "risk_score": .99}], "1900-01-01")
    assert len(cs.queue(conn)) == 3 and len(cs.queue(conn, opened_for="2025-08-16")) == 1
    assert cs.find(conn, cs.SMOKE, "1900-01-01")["priority"] == "P1"

def test_history_columns_and_smoke(conn):
    cs.open_cases(conn, ROWS + [{"athlete_id": cs.SMOKE, "priority": "P1", "risk_score": .99}], "2025-08-15")
    c = cs.queue(conn)[0]
    cs.transition(conn, c["case_id"], "action_set", "coach", c["updated_at"], action="monitor", note="private note")
    h = cs.history(conn)
    assert not {"actor", "assigned_to", "note"} & set(h.columns) and sorted(h.athlete_id) == ["A01", "A02"]
    assert h.set_index("athlete_id").n_events.to_dict() == {"A01": 1, "A02": 0}

def test_token_renewed_after_ttl_and_hidden():
    t, minted = [0.0], iter(["tok1", "tok2"])
    tok = cs.Token(lambda: next(minted), ttl=2700, clock=lambda: t[0])
    cp = {}
    cs.password_hook(tok)(None, None, [], cp)
    assert cp["password"] == "tok1"
    t[0] = 2699; assert tok() == "tok1"
    t[0] = 2700; cs.password_hook(tok)(None, None, [], cp)
    assert cp["password"] == "tok2" and "tok2" in cs._HIDE and "tok1" not in cs._HIDE
    assert cs.safe("x tok2 y") == "x <redacted> y"

def test_safe_masks_before_hide():
    s = cs.safe(RuntimeError("at https://h.example/x for a@b.io via 10.1.2.3 on ep-1.cloud.databricks.com"))
    assert s == "at <url> for <principal> via <ip> on <host>"

def test_genie_space_ids_sorted_and_safe():
    import genie, space, json, re
    b = [(i, q, s) for i, (q, s) in enumerate(genie.BENCH, 1)]
    d = space.build_space("cat", {"high": .13, "elevated": .078, "model_version": "3"}, b)
    s, ids = json.dumps(d), []
    def walk(x):
        if isinstance(x, list) and x and all(isinstance(e, dict) and "id" in e for e in x):
            assert [e["id"] for e in x] == sorted(e["id"] for e in x)
        for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else []):
            walk(v)
        if isinstance(x, dict) and "id" in x: ids.append(x["id"])
    walk(d)
    assert len(ids) == len(set(ids)) == 26 and all(re.fullmatch("[0-9a-f]{32}", i) for i in ids)
    t = d["data_sources"]["tables"]
    assert [x["identifier"] for x in t] == sorted(x["identifier"] for x in t) and "armguard_gt" not in s and "club_il" not in s
    assert {"column_name": "facts_json", "exclude": True} in next(x for x in t if x["identifier"].endswith("briefs"))["column_configs"]
    assert len(d["benchmarks"]["questions"]) == 12 and not {q for _, q, _ in b} & {q for q, _ in space.EXAMPLES}
    assert "{club}" not in s and "{ops}" not in s and "`cat`.armguard_ops.pitcher_risk_current" in s

def test_genie_pitcher_rule_poll_pass_redact():
    import genie, genie_ops as ops, space
    assert all(space.PIT in genie.BENCH[i][1] for i in (6, 7)) and space.PIT in (R / "src/4_genai/enrich.py").read_text()
    t, seq = [0], iter(["NOT_STARTED", None, "SOMETHING_NEW", "DONE", "RUNNING"])
    tick = lambda s: t.__setitem__(0, t[0] + s)
    assert ops.poll(lambda: next(seq), str, clock=lambda: t[0], sleep=tick) == "DONE" and t[0] == 30
    assert ops.poll(lambda: "RUNNING", str, limit=60, clock=lambda: t[0], sleep=tick) == "RUNNING" and t[0] == 90
    assert ops.poll(lambda: "EVALUATION_FAILED", str, clock=lambda: t[0], sleep=tick) == "EVALUATION_FAILED"
    assert ops.passed(10, 12) and not ops.passed(9, 12) and not ops.passed(10, 10)
    s = ops.redact(RuntimeError("mkdirs /Workspace/Users/a.b@x.io/armguard on dbc-1.cloud.databricks.com: cat_x.t\nline2"), "cat_x")
    assert "@" not in s and "databricks.com" not in s and "cat_x" not in s and "line2" not in s

def test_app_buttons_are_playbook_and_legal_from_open():
    import ast
    src = (R / "src/7_app/app.py").read_text()
    acts = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ACTIONS")
    assert {a for _, _, a in acts} - {None} <= {"monitor", "limit_pitches", "extra_rest", "bullpen_only", "refer_medical"}
    assert all(to in cs.TRANSITIONS["open"] for _, to, _ in acts) and "text_area" not in src  # no free-text note input
    assert "Data © MLB Advanced Media (statsapi.mlb.com) · players pseudonymized · medical data synthetic" in src

def test_stable_folds_deterministic_balanced_shuffle_invariant():
    sizes = {"A03": 3, "A01": 3, "A02": 3, "A05": 2, "A04": 2, "A06": 1}
    g = pd.Series([a for a, n in sizes.items() for _ in range(n)])
    f = train.stable_folds(g)
    got = dict(zip(g, f))
    assert got == {"A01": 0, "A02": 1, "A03": 2, "A04": 3, "A05": 3, "A06": 0}
    load = np.bincount(f, minlength=4)
    assert load.tolist() == [4, 3, 3, 4]
    perm = np.random.RandomState(1).permutation(len(g))
    gs = g.iloc[perm].reset_index(drop=True)
    assert dict(zip(gs, train.stable_folds(gs))) == got
