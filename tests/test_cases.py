import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src/7_app"))
import pytest
from sqlalchemy import create_engine
import cases as cs

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
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src/6_genie"))
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
