"""t9: Genie Agent "ArmGuard Analyst": upsert the space, score 12 unseen benchmarks with the eval-run API -> eval_genie."""
import datetime as dt, json, os, sys, time
from space import CASES, DAILY, ILX, RISK, nid, sub

TITLE, METHOD = "ArmGuard Analyst", "genie_eval_run_api"
BENCH = [  # (question, reference SQL): frozen; never used as examples
    ("How many pitchers are in the arm-care queue at each priority level?",
     f"SELECT priority, COUNT(*) AS pitchers FROM {RISK} WHERE priority IS NOT NULL GROUP BY priority ORDER BY priority"),
    ("Which pitchers are priority P1?", f"SELECT display_name FROM {RISK} WHERE priority = 'P1'"),
    ("Who are the 5 pitchers with the highest injury risk score?", f"SELECT display_name, risk_score FROM {RISK} ORDER BY risk_score DESC LIMIT 5"),
    ("Which pitchers were down 1.5 mph or more in fastball velocity versus baseline in their last appearance on or before Aug 15, 2025?",
     "SELECT r.display_name, a.velo_delta FROM (SELECT athlete_id, velo_delta, ROW_NUMBER() OVER (PARTITION BY athlete_id ORDER BY game_date DESC) rn "
     "FROM {club}.gold_appearances WHERE game_date <= '2025-08-15') a JOIN {club}.silver_roster r ON a.athlete_id = r.athlete_id "
     "WHERE a.rn = 1 AND a.velo_delta <= -1.5 ORDER BY a.velo_delta"),
    ("How many pitches did each starting pitcher throw in the 7 days ending Aug 15, 2025?",
     f"SELECT display_name, pitches_7d FROM {DAILY} WHERE feature_date = '2025-08-15' AND role_sp = 1 ORDER BY pitches_7d DESC"),
    ("Which pitchers had an acute:chronic workload ratio above 1.5 on Aug 15, 2025?",
     f"SELECT display_name, acwr FROM {DAILY} WHERE feature_date = '2025-08-15' AND acwr > 1.5"),
    ("What are the 3 most common body parts in real 2025 MLB pitcher IL placements?",
     f"SELECT body_part, COUNT(*) AS placements FROM {ILX} WHERE position IN ('RHP', 'LHP') GROUP BY body_part ORDER BY placements DESC LIMIT 3"),
    ("How many real 2025 MLB pitcher IL placements were there?", f"SELECT COUNT(*) AS placements FROM {ILX} WHERE position IN ('RHP', 'LHP')"),
    ("What is the recommended action for each P2 pitcher?", f"SELECT display_name, recommended_action FROM {RISK} WHERE priority = 'P2'"),
    ("Which pitchers in the queue have a red-flag trainer note?",
     f"SELECT display_name FROM {RISK} WHERE note_flag = 'red_flag' AND priority IS NOT NULL"),
    ("How many arm-care cases are in each status?", f"SELECT status, COUNT(*) AS cases FROM {CASES} GROUP BY status ORDER BY status"),
    ("What is the average number of hours to the first decision on arm-care cases, by priority?",
     f"SELECT priority, AVG(hours_to_decision) AS avg_hours FROM {CASES} GROUP BY priority ORDER BY priority"),
]
PLANTED = (2, 4, 10, 11)  # answers fixed by the planted story + demo actions


def upsert(w, sp, c, wh, ser):
    """Update the space recorded in app_config, else create it. -> (space_id, verb)"""
    from databricks.sdk.errors import NotFound, PermissionDenied
    A = c.fq("armguard_ops", "app_config")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {A} (key STRING, value STRING)")
    sid = dict(sp.sql(f"SELECT key, value FROM {A}").collect()).get("genie_space_id")
    kw = {"title": TITLE, "serialized_space": ser, "description": "Cascadia Ospreys arm-care: risk queue, workload, briefs, cases, league IL."}
    try:
        if sid and w.genie.get_space(sid):
            return w.genie.update_space(sid, warehouse_id=wh, **kw).space_id or sid, "updated"
    except (NotFound, PermissionDenied):
        print("space gone: creating")
    parent = f"/Workspace/Users/{w.current_user.me().user_name}/armguard"  # never printed
    w.workspace.mkdirs(parent)
    sid = w.genie.create_space(wh, parent_path=parent, **kw).space_id
    sp.sql(f"MERGE INTO {A} t USING (SELECT 'genie_space_id' key, :v value) s ON t.key = s.key "
           "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *", args={"v": sid})
    return sid, "created"


def grant(w, sid):
    """CAN_RUN for the app SP (SPIKE Q9b) -> (log, ok); WAIT while the app is missing."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
    try:
        app_sp = w.apps.get("armguard").service_principal_client_id
    except NotFound:
        return "WAIT app 'armguard' not found (Task 10 re-runs t9)", True
    try:
        acl = [AccessControlRequest(service_principal_name=app_sp, permission_level=PermissionLevel.CAN_RUN)]
        w.permissions.update("genie", sid, access_control_list=acl)
        return "OK app SP: CAN_RUN", True
    except Exception as e:
        print("grant:", str(e)[:200])
        return f"FAIL {type(e).__name__}", False


def evaluate(w, sid):
    """Eval run over all benchmarks, <= 20 min (SPIKE Q9) -> (run, {question_id: (assessment, reasons)})."""
    g, val = w.genie, lambda x: getattr(x, "value", x)
    rid, t0 = g.genie_create_eval_run(sid).eval_run_id, time.time()
    while val((r := g.genie_get_eval_run(sid, rid)).eval_run_status) in ("NOT_STARTED", "RUNNING", None) and time.time() - t0 < 1200:
        time.sleep(10)
    if val(r.eval_run_status) != "DONE":
        raise SystemExit(f"eval run {rid}: {val(r.eval_run_status)}")
    res, tok, out = [], None, {}
    while True:
        page = g.genie_list_eval_results(sid, rid, page_token=tok)
        res, tok = res + (page.eval_results or []), page.next_page_token
        if not tok:
            break
    for x in res:
        d = g.genie_get_eval_result_details(sid, rid, x.result_id)
        out[x.benchmark_question_id] = (val(d.assessment), [val(a) for a in d.assessment_reasons or []])
        print("bench", x.benchmark_question_id, out[x.benchmark_question_id], [a.response for a in d.actual_response or []])
    return r, out


def main(cm, sx):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.version import __version__
    from pyspark.sql import SparkSession
    c, sp, w = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate(), WorkspaceClient()
    wh = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--warehouse_id=")), "")
    if not wh:
        raise SystemExit("--warehouse_id is required")
    print("databricks-sdk", __version__)
    tables = [x for x in sx.TABLES if sp.catalog.tableExists(f"`{c.catalog}`.{x}")]  # case_history: after the first snapshot
    bench = [(i, q, s) for i, (q, s) in enumerate(BENCH, 1) if "case_history" not in s or "armguard_ops.case_history" in tables]
    ref, bad = {}, []
    for i, _, s in bench:  # each reference SQL must run on live data
        try:
            ref[i] = [tuple(r) for r in sp.sql(sub(s, c.catalog)).collect()]
        except Exception as e:
            bad.append(f"Q{i}: {str(e)[:160]}")
    thr = sp.table(c.fq("armguard_ops", "model_thresholds")).first().asDict()
    sid, verb = upsert(w, sp, c, wh, json.dumps(sx.build_space(c.catalog, thr, bench, tables)))
    print("genie_space_id", sid)
    g, gok = grant(w, sid)
    run, res = evaluate(w, sid)
    rows = [(i, q, *res.get(nid(f"bench {i}"), ("MISSING", []))) for i, q, _ in bench]
    n_ok, n_rev = sum(r[2] == "GOOD" for r in rows), sum(r[2] == "NEEDS_REVIEW" for r in rows)
    ts = dt.datetime.now(dt.timezone.utc)
    sp.createDataFrame([(ts, METHOD, i, a == "GOOD") for i, _, a, _ in rows], "run_ts TIMESTAMP, method STRING, q_num INT, correct BOOLEAN") \
        .write.mode("append").saveAsTable(c.fq("armguard_ops", "eval_genie"))
    hero = sp.table(c.fq("armguard", "silver_roster")).where(f"athlete_id = '{cm.HERO}'").first()[1]
    fmt = lambda v: "; ".join(" ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) for r in v) or "(none)"
    tab = "\n".join(f"| {i} | {q[:72]} | {'Y' if a == 'GOOD' else 'N'} | {a} {' '.join(r)[:60]} |" for i, q, a, r in rows)
    plant = "\n".join(f"- Q{i}: {fmt(ref[i])}" for i in PLANTED if i in ref)
    cm.write_evidence(c, "t9_genie", f"Genie Agent '{TITLE}': benchmark run",
        f"Space {sid} ({verb}), {len(tables)} tables; benchmarks unseen by Genie\nMethod: {METHOD} (Beta), correct = assessment GOOD\n"
        f"Score: {n_ok}/{len(rows)} (target 80%: {'PASS' if n_ok >= .8 * len(rows) else 'MISS'}); needs_review {n_rev} (not correct); "
        f"run counts {run.num_correct}/{run.num_questions}, review {run.num_needs_review}\n\n"
        f"| Q | question | correct | assessment |\n|---|---|---|---|\n{tab}\n\nPlanted answer sets (reference SQL via Spark):\n{plant}\n"
        f"Hero {cm.HERO} ({hero}) in Q2 and Q10: {'PASS' if all((hero,) in ref.get(i, []) for i in (2, 10)) else 'MISS'}\n"
        f"Reference SQL errors: {bad or 'none'}\nApp grant: {g}", limit=6000)
    if bad or not gok:
        raise SystemExit(f"reference SQL errors: {bad}; app grant: {g}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap()
    import space; main(common, space)
