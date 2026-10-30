"""t7: ai_extract IL parsing (league KPIs; club rows medical) and grounded, playbook-constrained ai_query briefs."""
import json, os, re, sys
import pandas as pd

DATES = "2025-07-25 2025-08-01 2025-08-08 2025-08-22 2025-08-29 2025-09-05 2025-09-12 2025-09-19 2025-09-20".split()
PROMPT_V1 = ("You assist an MLB club's pitching coach and high-performance staff. Use ONLY numbers that appear in the facts JSON. Never mention "
             "diagnoses or medical details. Do not propose specific pitch counts; staff set limits. recommended_action must be exactly one of: monitor, "
             "limit_pitches, extra_rest, bullpen_only, refer_medical. Reply with ONLY a single JSON object with exactly the keys headline, why, "
             "recommended_action, confidence_note. Facts: ")
E = lambda *v: {"type": "enum", "labels": list(v)}
SCHEMA = json.dumps({"player": {"type": "string"}, "position": {"type": "string"}, "il_type": E("7-day IL", "10-day IL", "15-day IL", "60-day IL"),
                     "body_part": E("elbow", "shoulder", "forearm", "back", "oblique", "hamstring", "other"), "side": E("left", "right", "unknown"),
                     "injury": {"type": "string"}})
FACTS = "display_name tier priority note_flag risk_score velo_delta pitches_7d acwr days_since_app top_drivers".split()
FACT_SQL = {"risk_score": "round(risk_score, 2)", "velo_delta": "round(velo_delta, 1)", "acwr": "round(acwr, 2)",
            "pitches_7d": "int(pitches_7d)", "days_since_app": "int(days_since_app)"}
MED = ("body_part", "side", "injury")


def parse_json_obj(text):
    """LLM text -> dict: drop prose/markdown fences around the outermost {...}; anything unparseable -> None."""
    try:
        o = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return o if isinstance(o, dict) else None
    except (TypeError, ValueError, AttributeError):
        return None


def extracted(c, table, where, keys, fields):
    """ai_extract v2.1 over IL wire text; fields read with the SPIKE Q7 path x:response:<field>:value."""
    cols = ", ".join(f"x:response:{f}:value::string {f}" for f in fields)
    return (f"SELECT {keys}, {cols} FROM (SELECT *, ai_extract(description, :schema, MAP('version', '2.1')) x "
            f"FROM {c.fq('armguard', table)} WHERE {where})")


def extract(sp, c):
    a = {"schema": SCHEMA}
    sp.sql(f"CREATE OR REPLACE TABLE {c.fq('armguard_ops', 'il_extractions')} AS " + extracted(  # league rows: no description, no injury
        c, "silver_transactions", "lower(description) LIKE '%placed%'", "string(txn_id) event_id", ("player", "position", "il_type") + MED[:2]), args=a)
    t = c.fq("armguard", "club_il_extractions")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {t} (event_id STRING, athlete_id STRING, body_part STRING, side STRING, injury STRING)")
    abac = "mask_medical" in str(sp.sql(f"SHOW POLICIES ON SCHEMA `{c.catalog}`.armguard").collect())
    for k in MED:  # tag before any row lands: the schema ABAC policy masks tagged columns; legacy path also needs SET MASK
        sp.sql(f"ALTER TABLE {t} ALTER COLUMN {k} SET TAGS ('armguard_sensitivity' = 'medical')")
        if not abac:
            sp.sql(f"ALTER TABLE {t} ALTER COLUMN {k} SET MASK `{c.catalog}`.armguard_gov.mask_medical")
    sp.sql(f"INSERT OVERWRITE {t} " + extracted(c, "silver_il_events", "action = 'placed'", "event_id, athlete_id", MED), args=a)
    return abac


def briefs(sp, c, cm, sc):
    P = c.fq("armguard_ops", "pitcher_risk_current")
    cur = sp.sql(f"SELECT p.*, g.days_since_app FROM {P} p JOIN {c.fq('armguard', 'gold_pitcher_daily_features')} g "
                 "ON g.athlete_id = p.athlete_id AND g.feature_date = p.as_of_date").toPandas()
    df = pd.concat([cur] + [sc.score_date(sp, c, d) for d in DATES]).dropna(subset=["priority"])
    if df.empty:
        raise SystemExit("no prioritized pitchers on any brief date")
    sp.createDataFrame(df[["athlete_id", "as_of_date", *FACTS]]).createOrReplaceTempView("brief_rows")
    ns = ", ".join(f"'{k}', {FACT_SQL.get(k, k)}" for k in FACTS)
    sp.sql(f"CREATE OR REPLACE TEMP VIEW brief_inputs AS SELECT athlete_id, as_of_date d, "
           f"to_json(named_struct({ns}), map('ignoreNullFields', 'false')) facts_json FROM brief_rows")
    out = sp.sql(f"SELECT athlete_id, d, facts_json, ai_query('{c.llm_endpoint}', concat(:prompt, facts_json), failOnError => false) r "
                 "FROM brief_inputs", args={"prompt": PROMPT_V1}).selectExpr("athlete_id", "d", "facts_json", "r.result t", "r.errorMessage err").toPandas()
    return pd.DataFrame([{"athlete_id": r.athlete_id, "as_of_date": r.d, **cm.finalize_brief(parse_json_obj(r.t), json.loads(r.facts_json)),
                          "prompt_version": "v1", "facts_json": r.facts_json} for r in out.itertuples()]), out.err.notna().sum()


def main(cm):
    import score as sc
    from pyspark.sql import SparkSession
    c, sp = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate()
    if not re.fullmatch(r"databricks-[a-z0-9-]+", c.llm_endpoint):  # the endpoint must be a SQL literal
        raise SystemExit(f"invalid llm_endpoint {c.llm_endpoint!r}")
    abac = extract(sp, c)
    b, errs = briefs(sp, c, cm, sc)
    P, B, IL = (c.fq("armguard_ops", t) for t in ("pitcher_risk_current", "pitcher_briefs", "il_extractions"))
    sp.createDataFrame(b).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(B)
    sp.sql(f"MERGE INTO {P} t USING (SELECT * FROM {B} WHERE as_of_date = DATE(:d)) s ON t.athlete_id = s.athlete_id "
           "WHEN MATCHED THEN UPDATE SET t.headline = s.headline, t.recommended_action = s.recommended_action", args={"d": c.as_of_date})
    k = sp.sql(f"SELECT coalesce(body_part, 'unspecified') body_part, count(*) placements FROM {IL} WHERE position IN ('RHP', 'LHP') "
               "GROUP BY 1 ORDER BY 2 DESC").toPandas()
    h = b[(b.athlete_id == "A22") & (b.as_of_date.astype(str) == c.as_of_date)][["headline", "recommended_action", "fallback"]]
    cm.write_evidence(c, "t7_genai", "GenAI: IL extraction KPIs and pitcher briefs",
        f"Real 2025 MLB pitcher IL placements by body part (ai_extract over {sp.table(IL).count()} league placements):\n"
        f"{k.head(6).to_markdown(index=False)}\nTotal pitcher IL placements: {k.placements.sum()}\n"
        f"Club IL rows: {sp.table(c.fq('armguard', 'club_il_extractions')).count()} (medical; masked via {'ABAC policy' if abac else 'SET MASK'})\n\n"
        f"Briefs: {len(b)} over {b.as_of_date.nunique()} dates; fallback {b.fallback.sum()} ({b.fallback.mean():.0%}); ai_query errors {errs}\n"
        f"Hero A22 brief {c.as_of_date}:\n{h.to_markdown(index=False)}")


if __name__ == "__main__":
    R = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path[:0] = [R, R + "/3_ml"]
    import common; common.bootstrap(); main(common)
