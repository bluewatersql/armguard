"""Lakebase case store: transactional case queue + audit trail (app, t8, ops tools)."""
import datetime as dt, re, threading, time
from sqlalchemy import text

STATUSES = ("open", "acknowledged", "action_set", "referred", "cleared", "placed_il")
TRANSITIONS = {"open": {"acknowledged", "action_set", "referred", "cleared"}, "acknowledged": {"action_set", "referred", "cleared"},
               "action_set": {"referred", "cleared", "placed_il"}, "referred": {"cleared", "placed_il"}, "cleared": set(), "placed_il": set()}
class StaleUpdate(Exception): ...
class IllegalTransition(Exception): ...
T = lambda s, n: f"{s}.{n}" if s else n
now = lambda: dt.datetime.now(dt.timezone.utc)
day = lambda d: dt.date.fromisoformat(str(d)[:10])  # str or date -> date (Postgres DATE params)
SMOKE = "SMOKE"  # ops-tools smoke case: never in the queue or the snapshot
_HIDE = []  # host, user, current token: redacted from printed errors
_MASK = ((r"https?://\S+", "<url>"), (r"[\w.-]+\.databricks\.com\b", "<host>"), (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>"),
         (r"\S+@\S+", "<principal>"))

def safe(e):
    s = e if isinstance(e, str) else (str(e).splitlines() or [type(e).__name__])[0][:200]
    for h in _HIDE:
        s = s.replace(h, "<redacted>")
    for p, r in _MASK:  # also covers errors raised before _HIDE is filled
        s = re.sub(p, r, s)
    return s

class Token:
    """Cached OAuth DB credential, re-minted after ttl seconds (tokens last ~1 h); thread-safe."""
    def __init__(self, mint, ttl=45 * 60, clock=time.monotonic):
        self.mint, self.ttl, self.clock, self.tok, self.t, self.lock = mint, ttl, clock, None, 0.0, threading.Lock()

    def __call__(self):
        with self.lock:
            if self.tok is None or self.clock() - self.t >= self.ttl:
                old, self.tok, self.t = self.tok, self.mint(), self.clock()
                _HIDE[:] = [h for h in _HIDE if h != old] + [self.tok]
            return self.tok

def password_hook(tok):
    """SQLAlchemy do_connect listener: every new DB connection gets a fresh-enough token."""
    return lambda dialect, rec, cargs, cparams: cparams.__setitem__("password", tok())

def connect(w, branch, db="databricks_postgres"):
    """Engine for the caller's identity on the branch's READ_WRITE endpoint (OAuth DB credential); never print the URL."""
    from sqlalchemy import URL, create_engine, event
    for _ in range(30):  # a new project's primary endpoint may still be starting
        ep = next((e for e in w.postgres.list_endpoints(branch) if "READ_WRITE" in str(e.status.endpoint_type)), None)
        if ep and ep.status.hosts and ep.status.hosts.host:
            break
        time.sleep(10)
    else:
        raise RuntimeError("no READ_WRITE endpoint with a host on the branch")
    user, host = w.current_user.me().user_name, ep.status.hosts.host
    _HIDE[:] = list(dict.fromkeys(_HIDE + [host, user]))
    url = URL.create("postgresql+psycopg", username=user, host=host, port=5432, database=db, query={"sslmode": "require"})  # no password
    eng = create_engine(url, pool_pre_ping=True, pool_recycle=1800, connect_args={"connect_timeout": 30})
    event.listen(eng, "do_connect", password_hook(Token(lambda: w.postgres.generate_database_credential(endpoint=ep.name).token)))
    return eng

def ddl(s="ops"):
    pk, ts = ("BIGSERIAL PRIMARY KEY", "TIMESTAMPTZ") if s else ("INTEGER PRIMARY KEY AUTOINCREMENT", "TIMESTAMP")
    return ([f"CREATE SCHEMA IF NOT EXISTS {s}"] if s else []) + [
        f"""CREATE TABLE IF NOT EXISTS {T(s,'cases')} (case_id {pk}, athlete_id TEXT NOT NULL, opened_for DATE NOT NULL,
        priority TEXT NOT NULL, risk_score DOUBLE PRECISION, status TEXT NOT NULL DEFAULT 'open' CHECK (status IN
        ('open','acknowledged','action_set','referred','cleared','placed_il')), assigned_to TEXT, action TEXT,
        pitch_limit INTEGER, created_at {ts} NOT NULL, updated_at {ts} NOT NULL, UNIQUE (athlete_id, opened_for))""",
        f"""CREATE TABLE IF NOT EXISTS {T(s,'case_events')} (event_id {pk}, case_id BIGINT NOT NULL REFERENCES
        {T(s,'cases')}(case_id), actor TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL, note TEXT, ts {ts} NOT NULL)"""]

def open_cases(c, rows, d, schema=None):
    n = 0
    for r in rows:
        s = r.get("risk_score")
        n += c.execute(text(f"""INSERT INTO {T(schema,'cases')} (athlete_id, opened_for, priority, risk_score, status, created_at,
            updated_at) VALUES (:a,:d,:p,:s,'open',:n,:n) ON CONFLICT (athlete_id, opened_for) DO NOTHING"""),
            {"a": r["athlete_id"], "d": day(d), "p": r["priority"], "s": None if s is None else float(s), "n": now()}).rowcount or 0
    return n

def queue(c, schema=None, opened_for=None):
    q = f"SELECT * FROM {T(schema,'cases')} WHERE athlete_id<>:s" + (" AND opened_for=:d" if opened_for else "")
    p = {"s": SMOKE, "d": day(opened_for) if opened_for else None}
    return [dict(r) for r in c.execute(text(q + " ORDER BY priority, risk_score DESC"), p).mappings()]

def find(c, athlete_id, d, schema=None):
    q = f"SELECT * FROM {T(schema,'cases')} WHERE athlete_id=:a AND opened_for=:d"
    return dict(c.execute(text(q), {"a": athlete_id, "d": day(d)}).mappings().one())

def events(c, cid, schema=None):
    return [dict(r) for r in c.execute(text(f"SELECT * FROM {T(schema,'case_events')} WHERE case_id=:c ORDER BY event_id"), {"c": cid}).mappings()]

def transition(c, cid, to, actor, expected, action=None, pitch_limit=None, note=None, schema=None):
    row = c.execute(text(f"SELECT status, CASE WHEN updated_at=:e THEN 1 ELSE 0 END fresh FROM {T(schema,'cases')} WHERE case_id=:c"),
                    {"c": cid, "e": expected}).one_or_none()  # same comparison as the UPDATE below
    if row is None:
        raise LookupError(f"case {cid} not found")
    cur = row[0]
    if not row[1]:  # staleness first: a stale caller must reload, whatever move it attempts
        raise StaleUpdate(f"case {cid} changed since you loaded it; reload and retry")
    if to not in TRANSITIONS[cur]:
        raise IllegalTransition(f"{cur} -> {to} is not allowed")
    n = now()  # optimistic lock again in the UPDATE: closes the race after the read above
    res = c.execute(text(f"""UPDATE {T(schema,'cases')} SET status=:s, action=coalesce(:a,action), pitch_limit=coalesce(:l,pitch_limit),
        assigned_to=:u, updated_at=:n WHERE case_id=:c AND updated_at=:e AND status=:f"""),
        {"s": to, "a": action, "l": pitch_limit, "u": actor, "n": n, "c": cid, "e": expected, "f": cur})
    if res.rowcount != 1:
        raise StaleUpdate(f"case {cid} changed since you loaded it; reload and retry")
    c.execute(text(f"INSERT INTO {T(schema,'case_events')} (case_id, actor, from_status, to_status, note, ts) VALUES (:c,:u,:f,:t,:o,:n)"),
              {"c": cid, "u": actor, "f": cur, "t": to, "o": note, "n": n})
    print(f"CASE_UPDATE case_id={cid} from={cur} to={to} action={action}")
    return dict(c.execute(text(f"SELECT * FROM {T(schema,'cases')} WHERE case_id=:c"), {"c": cid}).mappings().one())

def _ids(c, where, p, schema):
    return [r[0] for r in c.execute(text(f"SELECT case_id FROM {T(schema,'cases')} WHERE {where}"), p)]

def delete_cases(c, athlete_id, d, schema=None):
    ids = _ids(c, "athlete_id=:a AND opened_for=:d", {"a": athlete_id, "d": day(d)}, schema)
    for i in ids:
        c.execute(text(f"DELETE FROM {T(schema,'case_events')} WHERE case_id=:c"), {"c": i})
        c.execute(text(f"DELETE FROM {T(schema,'cases')} WHERE case_id=:c"), {"c": i})
    return len(ids)

def reset_cases(c, d, schema=None):
    ids = _ids(c, "opened_for=:d AND status<>'open'", {"d": day(d)}, schema)
    for i in ids:
        c.execute(text(f"DELETE FROM {T(schema,'case_events')} WHERE case_id=:c"), {"c": i})
        c.execute(text(f"UPDATE {T(schema,'cases')} SET status='open', action=NULL, pitch_limit=NULL, assigned_to=NULL, updated_at=:n "
                       "WHERE case_id=:c"), {"n": now(), "c": i})
    return len(ids)

def history(c, schema=None):
    """One row per case (SMOKE excluded) with n_events, first decision and hours to it; never actor / assigned_to / note."""
    import pandas as pd
    k = "c.case_id, c.athlete_id, c.opened_for, c.priority, c.risk_score, c.status, c.action, c.pitch_limit, c.created_at, c.updated_at"
    h = pd.read_sql(text(f"SELECT {k}, count(e.event_id) n_events, min(e.ts) decided_at FROM {T(schema,'cases')} c LEFT JOIN "
                         f"{T(schema,'case_events')} e ON e.case_id = c.case_id WHERE c.athlete_id <> :s GROUP BY {k}"), c, params={"s": SMOKE})
    for t in ("created_at", "updated_at", "decided_at"):
        h[t] = pd.to_datetime(h[t], utc=True).dt.tz_localize(None)
    h["hours_to_decision"] = (h.decided_at - h.created_at).dt.total_seconds() / 3600
    return h.astype({"pitch_limit": "Int64", "n_events": "Int64", "risk_score": float})
