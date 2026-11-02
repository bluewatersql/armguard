"""t9: Genie Agent "ArmGuard Analyst": upsert the space, score 12 unseen benchmarks with the eval-run API -> eval_genie."""
import datetime as dt, json, os, sys, traceback
import genie_ops as ops, space as sx
from space import CASES, DAILY, ILX, PIT, RISK

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
     f"SELECT body_part, COUNT(*) AS placements FROM {ILX} WHERE {PIT} GROUP BY body_part ORDER BY placements DESC LIMIT 3"),
    ("How many real 2025 MLB pitcher IL placements were there?", f"SELECT COUNT(*) AS placements FROM {ILX} WHERE {PIT}"),
    ("What is the recommended action for each P2 pitcher?", f"SELECT display_name, recommended_action FROM {RISK} WHERE priority = 'P2'"),
    ("Which pitchers in the queue have a red-flag trainer note?",
     f"SELECT display_name FROM {RISK} WHERE note_flag = 'red_flag' AND priority IS NOT NULL"),
    ("How many arm-care cases are in each status?", f"SELECT status, COUNT(*) AS cases FROM {CASES} GROUP BY status ORDER BY status"),
    ("What is the average number of hours to the first decision on decided arm-care cases, by priority?",
     f"SELECT priority, ROUND(AVG(hours_to_decision), 1) AS avg_hours FROM {CASES} WHERE hours_to_decision IS NOT NULL GROUP BY priority ORDER BY priority"),
]
PLANTED = (2, 4, 10, 11)  # answers fixed by the planted story + demo actions


def main(cm, c):
    """-> exit message (written after evidence) or None."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.version import __version__
    from pyspark.sql import SparkSession, functions as F
    sp, w = SparkSession.builder.getOrCreate(), WorkspaceClient()
    wh = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--warehouse_id=")), "")
    if not wh:
        raise SystemExit("--warehouse_id is required")
    print("databricks-sdk", __version__)
    tables = [x for x in sx.TABLES if sp.catalog.tableExists(f"`{c.catalog}`.{x}")]
    bench = [(i, q, s) for i, (q, s) in enumerate(BENCH, 1) if "case_history" not in s or "armguard_ops.case_history" in tables]
    ref, bad = {}, []
    for i, _, s in bench:
        try:
            ref[i] = [tuple(r) for r in sp.sql(sx.sub(s, c.catalog)).collect()]
        except Exception as e:
            bad.append(f"Q{i}: {ops.redact(e, c.catalog)}")
    thr = sp.table(c.fq("armguard_ops", "model_thresholds")).first().asDict()
    sid, verb = ops.upsert(w, sp, c, wh, json.dumps(sx.build_space(c.catalog, thr, bench, tables)), TITLE)
    print("genie_space_id", sid)
    g, gok = ops.grant(w, sid)
    err = f"reference SQL errors: {bad}; app grant: {g}" if bad or not gok else None
    run, res, fail = ops.try_evaluate(w, sid, g, c.catalog)
    rows = [(i, q, *res.get(sx.nid(f"bench {i}"), ("MISSING", []))) for i, q, _ in bench]
    n_ok, n_rev = sum(r[2] == "GOOD" for r in rows), sum(r[2] == "NEEDS_REVIEW" for r in rows)
    ts = dt.datetime.now(dt.timezone.utc)
    out = [(ts, METHOD + " FAILED", 0, None)] if fail else [(ts, METHOD, i, a == "GOOD") for i, _, a, _ in rows]  # q_num 0 = whole run
    sp.createDataFrame(out, "run_ts TIMESTAMP, method STRING, q_num INT, correct BOOLEAN") \
        .write.mode("append").saveAsTable(c.fq("armguard_ops", "eval_genie"))
    if fail:  # fail-soft once upserted: t10_genai_eval still runs; a failed grant or reference SQL still fails t9
        cm.write_evidence(c, "t9_genie", f"Genie Agent '{TITLE}': eval run FAILED", f"Space {sid} ({verb}), {len(tables)} tables\n{fail}\n"
                          f"Reference SQL errors: {bad or 'none'}\nApp grant: {g}")
        return err
    h = sp.table(c.fq("armguard", "silver_roster")).where(F.col("athlete_id") == cm.HERO).first()
    hero = h["display_name"] if h else None
    fmt = lambda v: "; ".join(" ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) for r in v) or "(none)"
    tab = "\n".join(f"| {i} | {q[:72]} | {'Y' if a == 'GOOD' else 'N'} | {a} {' '.join(r)[:60]} |" for i, q, a, r in rows)
    plant = "\n".join(f"- Q{i}: {fmt(ref[i])}" for i in (*PLANTED, 8) if i in ref)
    cm.write_evidence(c, "t9_genie", f"Genie Agent '{TITLE}': benchmark run",
        f"Space {sid} ({verb}), {len(tables)} tables; benchmarks unseen by Genie\nMethod: {METHOD} (Beta), correct = assessment GOOD\n"
        f"Score: {n_ok}/{len(rows)} (target >= 10/12: {'PASS' if ops.passed(n_ok, len(rows)) else 'MISS'}); needs_review {n_rev} (not correct); "
        f"run {run.num_correct}/{run.num_questions}\n\n"
        f"| Q | question | correct | assessment |\n|---|---|---|---|\n{tab}\n\nPlanted answer sets (reference SQL via Spark; Q8 = t7 KPI):\n{plant}\n"
        f"Hero {cm.HERO} ({hero}) in Q2 and Q10: {'PASS' if hero and all((hero,) in ref.get(i, []) for i in (2, 10)) else 'MISS'}\n"
        f"Reference SQL errors: {bad or 'none'}\nApp grant: {g}", limit=6000)
    return err


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap()
    c = common.Cfg.from_args(sys.argv[1:])
    try:
        err = main(common, c)
    except BaseException as e:  # FAILED evidence; redacted log
        err = ops.redact(str(e) if isinstance(e, SystemExit) else traceback.format_exc(), c.catalog)
        print(err)
        common.write_evidence(c, "t9_genie", f"Genie Agent '{TITLE}': FAILED", err.strip().splitlines()[-1])
        raise SystemExit("t9 failed (redacted error above)") from None
    if err:
        raise SystemExit(err)
