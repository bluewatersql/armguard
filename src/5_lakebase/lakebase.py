"""Lakebase ops as the project owner: --mode=sync (t8) | snapshot | smoke | reset_demo."""
import os, re, sys, time, traceback

PROJ, BR, PGDB = "armguard", "projects/armguard/branches/production", "databricks_postgres"
GRANTS = {"ops": ("USAGE ON SCHEMA ops", "SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ops", "USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ops"),
          "armguard_ops": ("USAGE ON SCHEMA armguard_ops", "SELECT ON ALL TABLES IN SCHEMA armguard_ops")}  # app SP
_E = {}  # cached owner engine


def ensure_project(w):
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.postgres import Project
    try:
        return w.postgres.get_project(f"projects/{PROJ}") and "exists"
    except NotFound:
        cu = {"autoscaling_limit_min_cu": 0.5, "autoscaling_limit_max_cu": 2.0}
        p = Project.from_dict({"spec": {"display_name": "ArmGuard", "pg_version": 17, "default_endpoint_settings": cu}})
        w.postgres.create_project(project=p, project_id=PROJ).wait()
        return "created"


def connect(w):
    """Owner engine on the production branch (auto-renewed OAuth token; URL never printed)."""
    return cs.connect(w, BR, PGDB)


def tx(w, f):
    """f(conn) in its own transaction on the cached owner engine."""
    if "e" not in _E:
        _E["e"] = connect(w)
    with _E["e"].begin() as cx:
        return f(cx)


def sync_table(w, c):
    """Create the TRIGGERED synced table, or refresh its pipeline; wait up to 15 min. -> (state, ok)"""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.postgres import SyncedTable
    name, pid, upd = f"{c.catalog}.armguard_ops.pitcher_risk_synced", None, None
    get = lambda: w.postgres.get_synced_table(f"synced_tables/{name}")
    try:
        pid = (get().as_dict().get("status") or {}).get("pipeline_id")
    except NotFound:
        w.postgres.create_synced_table(synced_table=SyncedTable.from_dict({"spec": {
            "source_table_full_name": f"{c.catalog}.armguard_ops.pitcher_risk_current", "primary_key_columns": ["athlete_id"],
            "scheduling_policy": "TRIGGERED", "branch": BR, "postgres_database": PGDB,
            "new_pipeline_spec": {"storage_catalog": c.catalog, "storage_schema": "armguard_ops"}}}), synced_table_id=name)
    else:
        try:
            upd = w.pipelines.start_update(pid).update_id
        except Exception as e:  # no/deleted pipeline or update running: poll the table
            print("refresh:", cs.safe(e))
    t0, st = time.time(), "?"
    while time.time() - t0 < 900:
        time.sleep(15)
        try:
            x = getattr(w.pipelines.get_update(pid, upd).update, "state", None) if upd else getattr(get().status, "detailed_state", None)
            st = str(x).split(".")[-1]
        except NotFound:  # just created: not visible yet
            continue
        if st in ("COMPLETED", "CANCELED") or st.endswith(("_ONLINE", "NO_PENDING_UPDATE")) or "FAILED" in st:
            return f"{'refresh' if upd else 'sync'} {st}", "FAILED" not in st and st != "CANCELED"
    return f"timeout after 15 min ({st})", False


def app_sp(w):
    """App SP application id (validated) or None."""
    try:
        sp = w.apps.get("armguard").service_principal_client_id
    except Exception:
        return None
    return sp if sp and re.fullmatch(r"[0-9a-fA-F-]{36}", sp) else None


def grant(w):
    """-> (log, ok): WAIT if no app; with an app SP, any failed schema fails t8."""
    if not (sp := app_sp(w)):
        return "WAIT app 'armguard' not found", True
    bad = []
    for s, gs in GRANTS.items():  # a transaction per schema; sp validated above
        try:
            tx(w, lambda cx: [cx.exec_driver_sql(f'GRANT {g} TO "{sp}"') for g in gs])
        except Exception as e:
            bad.append(f"{s}: {cs.safe(e)}")
    return ("FAIL " + "; ".join(bad), False) if bad else ("OK app SP: ops rw, armguard_ops read", True)


def sync(c, w, sp):
    from sqlalchemy import text
    proj = ensure_project(w)
    tx(w, lambda cx: [cx.exec_driver_sql(s) for s in cs.ddl("ops")])
    state, ok = sync_table(w, c)
    (g, gok), P, A = grant(w), c.fq("armguard_ops", "pitcher_risk_current"), c.fq("armguard_ops", "app_config")
    rows = [r.asDict() for r in sp.sql(f"SELECT athlete_id, priority, risk_score, as_of_date FROM {P} WHERE priority IS NOT NULL").collect()]
    if {str(r["as_of_date"]) for r in rows} - {c.as_of_date}:
        raise SystemExit(f"pitcher_risk_current is not as of {c.as_of_date}")
    opened = tx(w, lambda cx: cs.open_cases(cx, rows, c.as_of_date, "ops"))
    total = len(tx(w, lambda cx: cs.queue(cx, "ops")))  # SMOKE excluded
    synced = tx(w, lambda cx: cx.execute(text("SELECT count(*) FROM armguard_ops.pitcher_risk_synced")).scalar_one()) if ok else "n/a"
    sp.sql(f"CREATE TABLE IF NOT EXISTS {A} (key STRING, value STRING)")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {c.fq('armguard_ops', 'case_history')} ({cm.HIST}, display_name STRING)")
    sp.sql(f"MERGE INTO {A} t USING (SELECT * FROM VALUES ('catalog', :cat), ('as_of_date', :d) AS v(key, value)) s ON t.key = s.key "
           "WHEN MATCHED THEN UPDATE SET t.value = s.value WHEN NOT MATCHED THEN INSERT *", args={"cat": c.catalog, "d": c.as_of_date})
    gold = sp.table(P).count()
    cm.write_evidence(c, "t8_lakebase", "Lakebase: synced risk table + transactional case store",
        f"Project {PROJ}: {proj}; pitcher_risk_synced: {state}\nSynced rows (Postgres) {synced} vs gold rows (Delta) {gold}: "
        f"{'PASS' if synced == gold else 'MISS'}\nCases opened this run {opened} (prioritized {len(rows)}); total {total}\nApp grants: {g}\n\n"
        "Why Lakebase: staff act on alerts in an app (row-level transactions, audit trail, optimistic locks); scores stay governed in Delta.")
    if not (ok and gok):
        raise SystemExit(f"synced table: {state}; app grants: {g}")


def snapshot(c, w, sp):
    import pandas as pd
    h = tx(w, lambda cx: cs.history(cx, "ops"))
    s = sp.createDataFrame(h[[k.split()[0] for k in cm.HIST.split(", ")]], cm.HIST)
    s = s.join(sp.table(c.fq("armguard", "silver_roster")).select("athlete_id", "display_name"), "athlete_id", "left")
    s.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(H := c.fq("armguard_ops", "case_history"))
    if sid := app_sp(w):  # t4 grants it too
        sp.sql(f"GRANT SELECT ON TABLE {H} TO `{sid}`")
    med = h.hours_to_decision.median()
    cm.write_evidence(c, "t11_snapshot", "Case history snapshot (Lakebase -> Delta)",
        f"{len(h)} cases -> armguard_ops.case_history (no actor/assignee columns)\nBy status: {h.status.value_counts().to_dict()}\n"
        f"Median hours to first decision: {'n/a (no decisions yet)' if pd.isna(med) else f'{med:.2f}'}")


def smoke(c, w):
    D, S, stale = "1900-01-01", cs.SMOKE, "MISS: stale update was accepted"
    tx(w, lambda cx: cs.delete_cases(cx, S, D, "ops"))  # leftover of an aborted run
    out = [f"opened {tx(w, lambda cx: cs.open_cases(cx, [{'athlete_id': S, 'priority': 'P1', 'risk_score': 0.99}], D, 'ops'))}"]
    try:
        c0 = tx(w, lambda cx: cs.find(cx, S, D, "ops"))

        def go(to, prev, who="coach", **k):  # one transaction per step
            return tx(w, lambda cx: cs.transition(cx, c0["case_id"], to, who, prev["updated_at"], schema="ops", **k))
        c1 = go("acknowledged", c0, "trainer")
        go("action_set", c1, action="limit_pitches")
        try:
            go("cleared", c1)  # c1 is stale now
        except cs.StaleUpdate:
            stale = "StaleUpdate caught"
        out += [f"event {e['event_id']}: {e['from_status']} -> {e['to_status']}" for e in tx(w, lambda cx: cs.events(cx, c0["case_id"], "ops"))]
    finally:
        k = tx(w, lambda cx: cs.delete_cases(cx, S, D, "ops"))
    cm.write_evidence(c, "t8b_smoke", "Lakebase smoke: transitions, optimistic lock, cleanup", "\n".join(out + [stale, f"deleted {k}"]))
    if stale.startswith("MISS") or k != 1:
        raise SystemExit("smoke failed")


def main():
    from databricks.sdk import WorkspaceClient
    c, w = cm.Cfg.from_args(sys.argv[1:]), WorkspaceClient()
    if c.mode in ("sync", "snapshot"):
        from pyspark.sql import SparkSession
        (sync if c.mode == "sync" else snapshot)(c, w, SparkSession.builder.getOrCreate())
    elif c.mode == "smoke":
        smoke(c, w)
    elif c.mode == "reset_demo":
        print(f"reset_demo: reopened {tx(w, lambda cx: cs.reset_cases(cx, c.as_of_date, 'ops'))} cases for {c.as_of_date}")
    else:
        raise SystemExit(f"unknown --mode={c.mode!r}")


if __name__ == "__main__":
    R = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path[:0] = [R, R + "/7_app"]
    import common as cm; cm.bootstrap()
    import cases as cs
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        print(cs.safe(traceback.format_exc()))
        raise SystemExit("lakebase task failed (redacted traceback above)") from None
