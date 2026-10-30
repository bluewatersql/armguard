"""ArmGuard app: Morning Board, Pitcher Detail, Ask Genie, Publish."""
import contextlib, datetime as dt, os, time
import pandas as pd, streamlit as st
st.set_page_config(page_title="ArmGuard · Cascadia Ospreys", layout="wide")
import cases
from sqlalchemy import text

CAT, JOB = os.environ.get("ARMGUARD_CATALOG", ""), os.environ.get("OPS_JOB_ID", "")
ATTRIBUTION = "Data © MLB Advanced Media (statsapi.mlb.com) · players pseudonymized · medical data synthetic"
ENV = ("ARMGUARD_CATALOG", "OPS_JOB_ID", "LAKEBASE_ENDPOINT", "DATABRICKS_WAREHOUSE_ID", "PGUSER")
ICON = {"P1": "🔴", "P2": "🟠", "P3": "⚪"}
DRIVERS = lambda r: (r["top_drivers"] or "n/a").replace("; ", "  \n")
ACTIONS = (("Acknowledge", "acknowledged", None), ("Set pitch limit", "action_set", "limit_pitches"),
           ("Extra rest", "action_set", "extra_rest"), ("Refer to medical", "referred", "refer_medical"), ("Clear", "cleared", None))


@st.cache_resource
def w():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


@st.cache_resource
def engine():
    return cases.connect(w(), "projects/armguard/branches/production")  # as the app SP


def q(sql, p=None):
    """Warehouse query as the app SP -> DataFrame."""
    from databricks import sql as dbsql
    c = w().config
    with dbsql.connect(server_hostname=c.host.split("//")[-1], credentials_provider=lambda: c.authenticate,
                       http_path="/sql/1.0/warehouses/" + os.environ["DATABRICKS_WAREHOUSE_ID"]) as cn, cn.cursor() as cur:
        cur.execute(sql, p or {})
        return pd.DataFrame([tuple(r) for r in cur.fetchall()], columns=[d[0] for d in cur.description])


def pg(sql, **p):
    with engine().connect() as c:
        return [dict(r) for r in c.execute(text(sql), p).mappings()]


@contextlib.contextmanager
def shown(what="Warehouse"):
    try:
        yield
    except Exception as e:  # not st.rerun/st.stop (BaseException)
        st.error(f"{what} error: {cases.safe(e)}")


@st.cache_resource
def boot():
    """Once per process; logs env var names only (never values, tokens, hosts or emails)."""
    print("APP_ENV present=" + ",".join(filter(os.environ.get, ENV)), flush=True)
    try:
        m = "value=" + str(q(f"SELECT note_text FROM {CAT}.armguard.silver_trainer_notes LIMIT 1").iat[0, 0])[:24]
    except Exception as e:
        m = f"error={type(e).__name__} {cases.safe(e)[:120]}"
    print("MASK_CHECK " + m, flush=True)


cached = st.cache_data(ttl=300, show_spinner=False)
config = cached(lambda: dict(q(f"SELECT key, value FROM {CAT}.armguard_ops.app_config").values.tolist()))
as_of = cached(lambda: pg("SELECT max(as_of_date) d FROM armguard_ops.pitcher_risk_synced")[0]["d"])


def board(d, active=True):
    return pg("SELECT c.*, display_name, tier, headline, recommended_action, top_drivers FROM ops.cases c JOIN armguard_ops."
              "pitcher_risk_synced USING (athlete_id) WHERE opened_for = :d" + " AND status NOT IN ('cleared', 'placed_il')" * active
              + " ORDER BY c.priority, c.risk_score DESC", d=d)


def act(r, to, action, label):
    """Button callback. r is the row as shown: its updated_at is the lock token; the note is fixed text."""
    limit = int(st.session_state[f"lim{r['case_id']}"]) if action == "limit_pitches" else None
    try:
        with engine().begin() as c:
            cases.transition(c, r["case_id"], to, ACTOR, r["updated_at"], action, limit, "playbook button: " + label, "ops")
        f = ("success", f"{r['display_name']}: {r['status']} → {to}")
    except Exception as e:
        f = {cases.StaleUpdate: ("warning", "Someone else just updated this case — reloaded."),
             cases.IllegalTransition: ("error", str(e))}.get(type(e), ("error", "Case store error: " + cases.safe(e)))
    st.session_state.flash = f


def morning():
    st.title(f"Morning Board · {AS_OF}")
    if f := st.session_state.pop("flash", None):
        getattr(st, f[0])(f[1])
    rows = board(AS_OF)
    if not rows:
        st.info(f"No open cases for {AS_OF}.")
    for r in rows:
        cid = r["case_id"]
        with st.container(border=True):
            st.subheader(f"{ICON.get(r['priority'], '⚪')} {r['priority']} · {r['display_name']} · {r['tier']}")
            st.write(r["headline"])
            st.caption(f"Recommended {r['recommended_action']} · **{r['status']}** {r['action'] or ''} {r['pitch_limit'] or ''}")
            st.expander("Top drivers").markdown(DRIVERS(r))
            b = st.columns([1, 1, 1.3, 1, 1.3, 1])
            b[1].number_input("Pitch limit", 40, 110, 75, 5, key=f"lim{cid}", label_visibility="collapsed")
            for col, (label, to, action) in zip((b[0], *b[2:]), ACTIONS):
                col.button(label, key=f"{label}{cid}", on_click=act, args=(r, to, action, label))


def detail():
    by = {x["athlete_id"]: x for x in board(AS_OF, active=False)}
    if not by:
        return st.info(f"No cases for {AS_OF}.")
    a = st.selectbox("Pitcher", by, format_func=lambda k: by[k]["display_name"])
    r, p = by[a], {"a": a, "d": AS_OF}
    st.title(f"{r['display_name']} · {r['tier']} · {r['priority']}")
    left, right = st.columns([3, 2])
    with left, shown():
        st.subheader("Velocity vs own baseline, 45 days")
        v = q(f"SELECT game_date, fb_velo, velo_base10 FROM {CAT}.armguard.gold_appearances WHERE athlete_id = :a "
              "AND game_date BETWEEN date_sub(:d, 45) AND :d ORDER BY game_date", p)
        st.line_chart(v.set_index("game_date").astype(float))
    with right, shown():
        st.subheader("Brief")
        b = q(f"SELECT headline, why, recommended_action, confidence_note FROM {CAT}.armguard_ops.pitcher_briefs "
              "WHERE athlete_id = :a AND as_of_date = :d", p)
        for h, why, ra, cn in b.values.tolist()[:1]:
            st.markdown(f"**{h}**\n\n{why}\n\n**Recommended action:** `{ra}`")
            st.caption(cn)
        if b.empty:
            st.info("No brief for this date.")
        st.markdown("**Top drivers**  \n" + DRIVERS(r))
    st.subheader("Last 3 trainer notes")
    with shown():
        st.dataframe(q(f"SELECT note_ts, author_role, note_text FROM {CAT}.armguard.silver_trainer_notes WHERE athlete_id = :a "
                       "AND note_ts < date_add(:d, 1) ORDER BY note_ts DESC LIMIT 3", p), hide_index=True)
    st.caption("Medical notes are masked by Unity Catalog unless you are medical staff.")


def genie():
    st.title("Ask Genie")
    sid = CFG.get("genie_space_id")
    if not sid:
        return st.warning("Genie space not configured yet: run t9_genie.")
    ques = st.text_input("Question", placeholder="Which pitchers lost the most velocity?").strip()
    if st.button("Ask", type="primary") and ques:
        with shown("Genie"), st.spinner("Genie is thinking..."):
            m = w().genie.start_conversation_and_wait(sid, ques, timeout=dt.timedelta(minutes=4))
            for at in m.attachments or []:
                if at.text and at.text.content:
                    st.markdown(at.text.content)
                if at.query:
                    st.code(at.query.query, language="sql")
                    s = w().genie.get_message_attachment_query_result(sid, m.conversation_id, m.message_id, at.attachment_id).statement_response
                    st.dataframe(pd.DataFrame((s.result and s.result.data_array) or [], columns=[c.name for c in s.manifest.schema.columns]))
            if not m.attachments:
                st.info("No answer; try rephrasing.")
    st.link_button("Open in Genie", f"{w().config.host.rstrip('/')}/genie/rooms/{sid}")


def publish():
    st.title("Publish case history to Delta")
    if not JOB.isdigit():
        return st.warning("OPS_JOB_ID not set.")
    if st.button("Publish snapshot", type="primary"):
        rid, ph = w().jobs.run_now(job_id=int(JOB), job_parameters={"mode": "snapshot"}).run_id, st.empty()
        for _ in range(180):  # 5 s polls, 15 min max
            s = w().jobs.get_run(rid).state
            ph.info(f"Run {rid}: {s.life_cycle_state.value} {s.result_state.value if s.result_state else ''}")
            if s.life_cycle_state.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                break
            time.sleep(5)


boot()
ACTOR = st.context.headers.get("X-Forwarded-Email", "staff")
try:
    CFG = config()
except Exception as e:
    CFG = {}
    st.warning("app_config unavailable: " + cases.safe(e))
P = {"Morning Board": morning, "Pitcher Detail": detail, "Ask Genie": genie, "Publish": publish}
qp = st.query_params.get("page")
page = st.sidebar.radio("ArmGuard", list(P), index=list(P).index(qp) if qp in P else 0)
try:
    AS_OF = as_of()  # the data's as-of date, never today
except Exception as e:
    AS_OF = CFG.get("as_of_date")
    st.sidebar.error("Case store unavailable: " + cases.safe(e))
st.sidebar.caption(f"Data as of {AS_OF} · Cascadia Ospreys")
with shown(page):
    P[page]()
st.divider()
st.caption(ATTRIBUTION)
