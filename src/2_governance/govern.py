"""Governance: --mode=setup (t0); --mode=govern (t4: tags, mask, PROBE, grants, lineage)."""
import os, re, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
import common; common.bootstrap()
from pyspark.sql import SparkSession

TAGS = {"armguard_sensitivity": ["medical", "public"], "armguard_tier": ["bronze", "silver", "gold"]}
MED = [("silver_trainer_notes", "note_text", "note_id"), ("silver_il_events", "description", "event_id")]
ST = {"silver_trainer_notes", "silver_il_events", "silver_pitches", "silver_wellness"}
GR = {"armguard": ["silver_roster", "silver_trainer_notes", "gold_appearances", "gold_pitcher_daily_features"],
      "armguard_ops": ["pitcher_risk_current", "pitcher_briefs", "il_extractions", "case_history", "model_thresholds", "app_config"],
      "armguard_gov": ["staff_roles"]}  # staff_roles lets the mask UDF lookup run for the SP (verify in Task 10)
WHO = ""


def _first(e):
    s = next((l.strip() for l in str(e).splitlines() if l.strip()), type(e).__name__)
    return re.sub(r"\S+@\S+", "<principal>", s.replace(WHO, "<principal>") if WHO else s)[:160]


def step(spark, sql, log, label, args=None):
    try:
        spark.sql(sql, args=args)
        log.append(f"OK   {label}")
        return True
    except Exception as e:
        log.append(f"SKIP {label}: {_first(e)}")
        return False


def tags(log):
    try:
        from databricks.sdk import WorkspaceClient, errors as E
        from databricks.sdk.service.tags import TagPolicy, Value
        w, dup = WorkspaceClient(), tuple(getattr(E, n) for n in ("AlreadyExists", "ResourceAlreadyExists") if hasattr(E, n))
        for k, vs in TAGS.items():
            try:
                try:
                    w.tag_policies.create_tag_policy(tag_policy=TagPolicy(tag_key=k, values=[Value(name=v) for v in vs]))
                    log.append(f"OK   governed tag {k} created")
                except dup:
                    cur = w.tag_policies.get_tag_policy(tag_key=k)
                    miss = set(vs) - {v.name for v in cur.values or []}
                    if miss:
                        cur.values = (cur.values or []) + [Value(name=v) for v in sorted(miss)]
                        w.tag_policies.update_tag_policy(tag_key=k, tag_policy=cur, update_mask="values")
                    log.append(f"OK   governed tag {k} exists" + (f", added {sorted(miss)}" if miss else ""))
            except Exception as e:
                log.append(f"SKIP governed tag {k}: {_first(e)}")
    except Exception as e:
        log.append(f"SKIP governed tags: {_first(e)}")


def setup(spark, cfg, log):
    G = f"`{cfg.catalog}`.armguard_gov"
    tags(log)
    step(spark, f"CREATE TABLE IF NOT EXISTS {G}.staff_roles (principal STRING, role STRING)", log, "staff_roles")
    step(spark, role_merge(G), log, "presenter=medical", {"p": cfg.presenter})
    step(spark, f"""CREATE OR REPLACE FUNCTION {G}.mask_medical(v STRING) RETURNS STRING
        RETURN CASE WHEN EXISTS (SELECT 1 FROM {G}.staff_roles r WHERE r.principal = current_user()
               AND r.role = 'medical') THEN v ELSE '[REDACTED - medical staff only]' END""", log, "mask UDF")


def role_merge(G):
    return (f"MERGE INTO {G}.staff_roles t USING (SELECT :p principal, 'medical' role) s ON t.principal = s.principal "
            "WHEN MATCHED THEN UPDATE SET role = 'medical' WHEN NOT MATCHED THEN INSERT *")


def probe(spark, C, who, log, name):
    qs = [f"SELECT {c} FROM {C}.armguard.{t} WHERE {c} IS NOT NULL ORDER BY {k} LIMIT 1" for t, c, k in MED]
    try:
        try:
            spark.sql(f"DELETE FROM {C}.armguard_gov.staff_roles WHERE principal = :p", args={"p": who})
            m = [spark.sql(q).first() for q in qs]
        finally:  # ALWAYS restore
            spark.sql(role_merge(f"{C}.armguard_gov"), args={"p": who})
        v = [spark.sql(q).first() for q in qs]
        if not all(m) or not all(v):
            r = ([], [], False, "no non-null row to probe")
        else:
            m, v = [x[0] for x in m], [x[0] for x in v]
            r = (m, v, all(x.startswith("[REDACTED") for x in m) and not any(x.startswith("[REDACTED") for x in v), "")
    except Exception as e:
        r = ([], [], False, _first(e))
    log.append(f"PROBE path {name} effective={r[2]}" + (f" ({r[3]})" if r[3] else ""))
    return r


def govern(spark, cfg, log):
    C = f"`{cfg.catalog}`"
    A, G = f"{C}.armguard", f"{C}.armguard_gov"
    for r in spark.sql(f"SHOW TABLES IN {A}").collect():
        t = r[1]
        if t.startswith("_"):
            continue
        kind = "STREAMING TABLE" if t in ST or t.startswith("bronze_") else "MATERIALIZED VIEW"
        lay = t.split("_")[0] if t.split("_")[0] in ("bronze", "gold") else "silver"
        step(spark, f"ALTER {kind} {A}.{t} SET TAGS ('armguard_tier'='{lay}')", log, f"tag {t} tier={lay}")
    for t, c, _ in MED:
        step(spark, f"ALTER STREAMING TABLE {A}.{t} ALTER COLUMN {c} SET TAGS ('armguard_sensitivity'='medical')", log, f"tag {t}.{c}")
    for t, c, _ in MED:  # path (b) must be tested alone
        step(spark, f"ALTER STREAMING TABLE {A}.{t} ALTER COLUMN {c} DROP MASK", log, f"clean mask {t}.{c}")
    step(spark, f"CREATE OR REPLACE POLICY mask_medical ON SCHEMA {A} COLUMN MASK {G}.mask_medical TO `account users` "
                f"FOR TABLES MATCH COLUMNS has_tag_value('armguard_sensitivity','medical') AS c ON COLUMN c", log, "ABAC policy")
    path, res = "b (ABAC policy)", probe(spark, C, cfg.presenter, log, "b")
    if not res[2] and not res[3].startswith("no non-null"):  # fallback
        dropped = step(spark, f"DROP POLICY IF EXISTS mask_medical ON SCHEMA {A}", log, "drop ABAC policy")
        path = "c (legacy SET MASK; ABAC policy " + ("dropped)" if dropped else "drop FAILED)")
        for t, c, _ in MED:
            step(spark, f"ALTER STREAMING TABLE {A}.{t} ALTER COLUMN {c} SET MASK {G}.mask_medical", log, f"mask {t}.{c}")
        res = probe(spark, C, cfg.presenter, log, "c")
    try:
        from databricks.sdk import WorkspaceClient
        sp = next((a.service_principal_client_id for a in WorkspaceClient().apps.list() if a.name == "armguard"), None)
    except Exception as e:
        sp = None
        log.append(f"SKIP apps.list: {_first(e)}")
    if sp:
        P = f"`{sp}`"
        step(spark, f"GRANT USE CATALOG ON CATALOG {C} TO {P}", log, "grant use catalog")
        for s, ts in GR.items():
            step(spark, f"GRANT USE SCHEMA ON SCHEMA {C}.{s} TO {P}", log, f"grant use schema {s}")
            for t in ts:
                step(spark, f"GRANT SELECT ON TABLE {C}.{s}.{t} TO {P}", log, f"grant select {s}.{t}")
        step(spark, f"GRANT EXECUTE ON FUNCTION {G}.mask_medical TO {P}", log, "grant execute mask_medical")
    else:
        log.append("WAIT app SP not found (re-run after Task 10)")
    for t, d in (("gold_appearances", "Pitcher appearance workload; own-baseline velocity deltas"),
                 ("gold_pitcher_daily_features", "Pitcher-day model features; label_il_14d")):
        step(spark, f"COMMENT ON TABLE {A}.{t} IS '{d}'", log, f"comment {t}")
    rows = {}
    for k, q, a in (("tags", f"SELECT table_name, column_name, tag_name, tag_value FROM {C}.information_schema.column_tags "
                             "WHERE schema_name='armguard' ORDER BY 1,2", None),
                    ("lin", "SELECT source_table_full_name, target_table_full_name, max(event_time) FROM system.access.table_lineage "
                            "WHERE split(target_table_full_name,'[.]')[0] = :cat AND split(target_table_full_name,'[.]')[1] LIKE 'armguard%' "
                            "GROUP BY ALL ORDER BY 3 DESC LIMIT 12", {"cat": cfg.catalog})):
        try:
            rows[k] = [" | ".join(map(str, r)) for r in spark.sql(q, args=a).collect()]
        except Exception as e:
            log.append(f"SKIP {k}: {_first(e)}")
    ok = bool(res[2])
    pv = "; ".join(f"{t}.{c} masked={m!r} visible(30)={v[:30]!r}" for (t, c, _), m, v in zip(MED, res[0], res[1])) + f"; effective={ok}" if res[0] else f"effective=False ({res[3]})"
    body = ("Step log:\n" + "\n".join(log) + f"\n\nMask mechanism that passed the probe: {path if ok else 'NONE'}\nProbe (presenter): {pv}"
            "\n\ncolumn_tags:\n" + "\n".join(rows.get("tags", [])) + "\n\nLineage:\n" + ("\n".join(rows.get("lin", [])) or "lineage not yet populated"))
    common.write_evidence(cfg, "t4_governance", "Governance: mask proven by probe", body, limit=6000)
    if not ok:
        raise SystemExit("mask NOT effective")


if __name__ == "__main__":
    cfg, log = common.Cfg.from_args(sys.argv[1:]), []
    WHO = cfg.presenter
    (setup if cfg.mode == "setup" else govern)(SparkSession.builder.getOrCreate(), cfg, log)
    if cfg.mode == "setup":
        print("\n".join(log))
