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
