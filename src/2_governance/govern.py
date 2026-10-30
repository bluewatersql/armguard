"""Governance: --mode=setup (t0) creates tags, roles table, mask UDF; --mode=govern (t4) tags, masks, PROBES, grants, lineage."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
import common; common.bootstrap()
from pyspark.sql import SparkSession

TAGS = {"armguard_sensitivity": ["medical", "public"], "armguard_tier": ["bronze", "silver", "gold"]}
MEDICAL = [("silver_trainer_notes", "note_text"), ("silver_il_events", "description")]
STREAMING = {"silver_trainer_notes", "silver_il_events", "silver_pitches", "silver_wellness"}
OPS_SELECT = ["pitcher_risk_current", "pitcher_briefs", "il_extractions", "case_history", "model_thresholds", "app_config"]
APP_SELECT = ["silver_roster", "silver_trainer_notes", "gold_appearances", "gold_pitcher_daily_features"]


def step(spark, sql, log, label, args=None):
    try:
        spark.sql(sql, args=args)
        log.append(f"OK   {label}")
        return True
    except Exception as e:
        log.append(f"SKIP {label}: {str(e).strip().splitlines()[0][:160]}")
        return False


def setup(spark, cfg, log):
    C, G = f"`{cfg.catalog}`", f"`{cfg.catalog}`.armguard_gov"
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.tags import TagPolicy, Value
        for k, vs in TAGS.items():
            try:
                WorkspaceClient().tag_policies.create_tag_policy(tag_policy=TagPolicy(tag_key=k, values=[Value(name=v) for v in vs]))
                log.append(f"OK   governed tag {k}")
            except Exception as e:
                log.append(("OK   " if "xist" in str(e) else "SKIP ") + f"governed tag {k}: {str(e).splitlines()[0][:120]}")
    except Exception as e:
        log.append(f"SKIP governed tags: {str(e).splitlines()[0][:120]}")
    step(spark, f"CREATE TABLE IF NOT EXISTS {G}.staff_roles (principal STRING, role STRING)", log, "staff_roles")
    step(spark, f"MERGE INTO {G}.staff_roles t USING (SELECT :p principal, 'medical' role) s ON t.principal = s.principal "
                f"WHEN NOT MATCHED THEN INSERT *", log, "presenter=medical", {"p": cfg.presenter})
    step(spark, f"""CREATE OR REPLACE FUNCTION {G}.mask_medical(v STRING) RETURNS STRING
        RETURN CASE WHEN EXISTS (SELECT 1 FROM {G}.staff_roles r WHERE r.principal = current_user()
               AND r.role = 'medical') THEN v ELSE '[REDACTED - medical staff only]' END""", log, "mask UDF")


def probe(spark, C, presenter):
    q = f"SELECT note_text FROM {C}.armguard.silver_trainer_notes LIMIT 1"
    spark.sql(f"DELETE FROM {C}.armguard_gov.staff_roles WHERE principal = :p", args={"p": presenter})
    try:
        masked = spark.sql(q).first()[0]
    finally:  # presenter row is ALWAYS restored
        spark.sql(f"INSERT INTO {C}.armguard_gov.staff_roles VALUES (:p, 'medical')", args={"p": presenter})
    visible = spark.sql(q).first()[0]
    return masked, visible, masked.startswith("[REDACTED") and not visible.startswith("[REDACTED")


def govern(spark, cfg, log):
    C = f"`{cfg.catalog}`"
    A, O, G = f"{C}.armguard", f"{C}.armguard_ops", f"{C}.armguard_gov"
    tabs = [r[1] for r in spark.sql(f"SHOW TABLES IN {A}").collect()]
    def alter(t):
        return "STREAMING TABLE" if t in STREAMING or t.startswith("bronze_") else "MATERIALIZED VIEW"
    for t in tabs:
        if t.startswith("_"):
            continue
        lay = "bronze" if t.startswith("bronze_") else "gold" if t.startswith("gold_") else "silver"
        step(spark, f"ALTER {alter(t)} {A}.{t} SET TAGS ('armguard_tier'='{lay}')", log, f"tag {t} tier={lay}")
    for t, c in MEDICAL:
        step(spark, f"ALTER STREAMING TABLE {A}.{t} ALTER COLUMN {c} SET TAGS ('armguard_sensitivity'='medical')", log, f"tag {t}.{c}")
    pol = (f"CREATE OR REPLACE POLICY mask_medical ON SCHEMA {A} COLUMN MASK {G}.mask_medical TO `account users` "
           f"FOR TABLES MATCH COLUMNS has_tag_value('armguard_sensitivity','medical') AS c ON COLUMN c")
    path = "b (ABAC policy)"
    step(spark, pol, log, "ABAC policy mask_medical")
    res = None
    try:
        res = probe(spark, C, cfg.presenter)
    except Exception as e:
        log.append(f"SKIP probe(b): {str(e).splitlines()[0][:160]}")
    log.append(f"PROBE path b effective={res[2] if res else False}")
    if not (res and res[2]):
        path = "c (ALTER COLUMN SET MASK)"
        for t, c in MEDICAL:
            step(spark, f"ALTER STREAMING TABLE {A}.{t} ALTER COLUMN {c} SET MASK {G}.mask_medical", log, f"mask {t}.{c}")
        try:
            res = probe(spark, C, cfg.presenter)
        except Exception as e:
            log.append(f"SKIP probe(c): {str(e).splitlines()[0][:160]}")
        log.append(f"PROBE path c effective={res[2] if res else False}")
    # grants
    sp = None
    try:
        from databricks.sdk import WorkspaceClient
        sp = next((a.service_principal_client_id for a in WorkspaceClient().apps.list() if a.name == "armguard"), None)
    except Exception as e:
        log.append(f"SKIP apps.list: {str(e).splitlines()[0][:120]}")
    if sp:
        P = f"`{sp}`"
        step(spark, f"GRANT USE CATALOG ON CATALOG {C} TO {P}", log, "grant use catalog")
        for s in ("armguard", "armguard_ops", "armguard_gov"):
            step(spark, f"GRANT USE SCHEMA ON SCHEMA {C}.{s} TO {P}", log, f"grant use schema {s}")
        for sch, ts in (("armguard", APP_SELECT), ("armguard_ops", OPS_SELECT)):
            for t in ts:
                step(spark, f"GRANT SELECT ON TABLE {C}.{sch}.{t} TO {P}", log, f"grant select {sch}.{t}")
        step(spark, f"GRANT SELECT ON TABLE {G}.staff_roles TO {P}", log, "grant select staff_roles")
        step(spark, f"GRANT EXECUTE ON FUNCTION {G}.mask_medical TO {P}", log, "grant execute mask_medical")
        step(spark, f"GRANT EXECUTE ON MODEL {O}.injury_risk TO {P}", log, "grant execute model injury_risk")
    else:
        log.append("WAIT app SP not found (re-run after Task 10)")
    # comments + lineage
    for t, d in (("gold_appearances", "Pitcher appearance workload with own-baseline velocity deltas"),
                 ("gold_pitcher_daily_features", "Pitcher-day model features; label_il_14d = IL placement in next 14 days")):
        step(spark, f"COMMENT ON TABLE {A}.{t} IS '{d}'", log, f"comment {t}")
    tags = lin = None
    try:
        tags = spark.sql(f"SELECT table_name, column_name, tag_name, tag_value FROM {C}.information_schema.column_tags "
                         f"WHERE schema_name='armguard' ORDER BY 1,2").collect()
    except Exception as e:
        log.append(f"SKIP column_tags: {str(e).splitlines()[0][:120]}")
    try:
        lin = spark.sql(f"SELECT source_table_full_name, target_table_full_name, max(event_time) FROM system.access.table_lineage "
                        f"WHERE target_table_full_name LIKE '{cfg.catalog}.armguard%' GROUP BY ALL ORDER BY 3 DESC LIMIT 12").collect()
    except Exception as e:
        log.append(f"SKIP lineage: {str(e).splitlines()[0][:120]}")
    ok = bool(res and res[2])
    pv = f"masked={res[0]!r}; visible(30)={res[1][:30]!r}; effective={ok}" if res else "probe did not run; effective=False"
    body = ("Step log:\n" + "\n".join(log) + f"\n\nMask path: {path}\nProbe (presenter): {pv}\n\ncolumn_tags:\n"
            + "\n".join(" | ".join(map(str, r)) for r in (tags or [])) + "\n\nLineage:\n"
            + ("\n".join(" -> ".join(str(x) for x in r) for r in lin) if lin else "lineage not yet populated"))
    common.write_evidence(cfg, "t4_governance", "Governance: mask proven by probe", body, limit=6000)
    if not ok:
        raise SystemExit("mask NOT effective")


if __name__ == "__main__":
    cfg, log = common.Cfg.from_args(sys.argv[1:]), []
    spark = SparkSession.builder.getOrCreate()
    (setup if cfg.mode == "setup" else govern)(spark, cfg, log)
    if cfg.mode == "setup":
        print("\n".join(log))
