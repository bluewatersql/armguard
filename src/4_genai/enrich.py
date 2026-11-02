"""t7: incremental ai_extract IL parsing (league KPIs; club rows medical), grounded ai_query briefs."""
import json, os, re, sys
from collections import Counter
import pandas as pd

DATES = "2025-07-25 2025-08-01 2025-08-08 2025-08-22 2025-08-29 2025-09-05 2025-09-12 2025-09-19 2025-09-20".split()
PROMPT_V1 = ("You assist an MLB club's pitching coach and high-performance staff. Use ONLY numbers that appear in the facts JSON. Never mention "
             "diagnoses or medical details. Do not propose specific pitch counts; staff set limits. recommended_action must be exactly one of: monitor, "
             "limit_pitches, extra_rest, bullpen_only, refer_medical. Reply with ONLY a single JSON object with exactly the keys headline, why, "
             "recommended_action, confidence_note. Facts: ")
E, S = lambda *v: {"type": "enum", "labels": list(v)}, {"type": "string"}
SCHEMA = json.dumps({"player": S, "position": S, "il_type": E("7-day IL", "10-day IL", "15-day IL", "60-day IL"), "body_part": E(
    "elbow", "shoulder", "forearm", "back", "oblique", "hamstring", "other"), "side": E("left", "right", "unknown"), "injury": S})
FACTS = "display_name tier priority note_flag risk_score velo_delta pitches_7d acwr days_since_app top_drivers".split()
FACT_SQL = {"risk_score": "round(risk_score, 2)", "velo_delta": "round(velo_delta, 1)", "acwr": "round(acwr, 2)",
            "pitches_7d": "int(pitches_7d)", "days_since_app": "int(days_since_app)"}
MED = ("body_part", "side", "injury")


def parse_json_obj(text):
    """Outermost {...} of LLM text (prose/fences ignored) -> dict; otherwise None."""
    try:
        o = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return o if isinstance(o, dict) else None
    except (TypeError, ValueError, AttributeError):
        return None


BRIEF_IN = "athlete_id STRING, as_of_date DATE, " + ", ".join(f"{k} {'DOUBLE' if k in FACT_SQL else 'STRING'}" for k in FACTS)
BRIEF_OUT = ("athlete_id STRING, as_of_date DATE, headline STRING, why STRING, recommended_action STRING, confidence_note STRING, "
             "fallback BOOLEAN, prompt_version STRING, facts_json STRING")


def endpoint(c):
    """ai_query needs a SQL literal endpoint: plain Databricks names only."""
    if not re.fullmatch(r"databricks-[a-z0-9-]+", c.llm_endpoint):
        raise SystemExit(f"invalid llm_endpoint {c.llm_endpoint!r}")
    return c.llm_endpoint


def extract_new(sp, c, src, target, key, where, cols, fields):
    """ai_extract v2.1 on rows not yet in target; errored rows are retried next run. -> (new, pending)"""
    vals = ", ".join(f"x:response:{f}:value::string {f}" for f in fields)  # ai_extract JSON
    new = f"FROM {c.fq('armguard', src)} s LEFT ANTI JOIN {target} e ON e.event_id = string(s.{key}) WHERE {where}"
    before = sp.table(target).count()
    sp.sql(f"""INSERT INTO {target} SELECT {cols}, {vals} FROM (
      SELECT s.*, ai_extract(s.description, :schema, MAP('version', '2.1')) x {new})
    WHERE x:error_message::string IS NULL AND x:response IS NOT NULL""", args={"schema": SCHEMA})
    return sp.table(target).count() - before, sp.sql(f"SELECT count(*) {new}").first()[0]


def extract(sp, c):
    L, t = c.fq("armguard_ops", "il_extractions"), c.fq("armguard", "club_il_extractions")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {L} (event_id STRING, player STRING, position STRING, il_type STRING, body_part STRING, side STRING)")
    league = extract_new(sp, c, "silver_transactions", L, "txn_id", "lower(s.description) LIKE '%placed%'",  # league: no description, no injury
                         "string(txn_id) event_id", ("player", "position", "il_type", "body_part", "side"))
    sp.sql(f"CREATE TABLE IF NOT EXISTS {t} (event_id STRING, athlete_id STRING, body_part STRING, side STRING, injury STRING)")
    try:
        pol = sp.sql(f"SHOW POLICIES ON SCHEMA `{c.catalog}`.armguard").collect()
        abac = any((p["Policy Name"], p["Policy Type"]) == ("mask_medical", "COLUMN_MASK") for p in pol)
    except Exception:  # ABAC not confirmed -> SET MASK below
        abac = False
    for k in MED:  # tagged before any row lands (ABAC masks tagged columns); no confirmed ABAC -> SET MASK
        sp.sql(f"ALTER TABLE {t} ALTER COLUMN {k} SET TAGS ('armguard_sensitivity' = 'medical')")
        if not abac:
            sp.sql(f"ALTER TABLE {t} ALTER COLUMN {k} SET MASK `{c.catalog}`.armguard_gov.mask_medical")
    club = extract_new(sp, c, "silver_il_events", t, "event_id", "s.action = 'placed'", "event_id, athlete_id", MED)
    return abac, league, club


def brief_rows(sp, c, sc):
    """Prioritized rows: demo date from pitcher_risk_current (t6), replay dates via score_date."""
    P = c.fq("armguard_ops", "pitcher_risk_current")
    cur = sp.sql(f"SELECT p.*, g.days_since_app FROM {P} p JOIN {c.fq('armguard', 'gold_pitcher_daily_features')} g "
                 "ON g.athlete_id = p.athlete_id AND g.feature_date = p.as_of_date").toPandas()
    got = sorted(set(cur.as_of_date.astype(str)))
    if got != [c.as_of_date]:
        raise SystemExit(f"pitcher_risk_current is as of {got or 'nothing'}, expected {c.as_of_date}")
    df = pd.concat([cur] + [sc.score_date(sp, c, d) for d in DATES if d != c.as_of_date]).dropna(subset=["priority"])
    if df.empty:
        raise SystemExit("no prioritized pitchers on any brief date")
    return df.astype(dict.fromkeys(FACT_SQL, float))


def finalize(cm, text, facts):
    """LLM brief only if parsed, in the playbook and grounded; else the template. -> (brief, fallback reason)"""
    raw = parse_json_obj(text)
    body = " ".join(str(raw.get(k) or "") for k in ("headline", "why", "confidence_note")) if raw else ""
    bad = [n for n in cm._NUM.findall(body) if not cm.numbers_grounded(n, facts)]
    b = cm.finalize_brief(None if bad else raw, facts)
    return b, f"ungrounded {bad}" if bad else "invalid" if b["fallback"] else ""


def briefs(sp, c, cm, sc):
    ep = endpoint(c)
    sc.sdf(sp, brief_rows(sp, c, sc), BRIEF_IN).createOrReplaceTempView("brief_rows")
    ns = ", ".join(f"'{k}', {FACT_SQL.get(k, k)}" for k in FACTS)
    sp.sql(f"CREATE OR REPLACE TEMP VIEW brief_inputs AS SELECT athlete_id, as_of_date d, "
           f"to_json(named_struct({ns}), map('ignoreNullFields', 'false')) facts_json FROM brief_rows")
    out = sp.sql(f"SELECT athlete_id, d, facts_json, ai_query('{ep}', concat(:prompt, facts_json), failOnError => false) r FROM brief_inputs",
                 args={"prompt": PROMPT_V1}).selectExpr("athlete_id", "d", "facts_json", "r.result t", "r.errorMessage err").toPandas()
    rows, why = [], Counter()
    for r in out.itertuples():
        b, reason = finalize(cm, r.t, json.loads(r.facts_json))
        if reason:
            why[reason.split()[0]] += 1
            print(f"fallback {r.athlete_id} {r.d}: {reason}")
        rows.append({"athlete_id": r.athlete_id, "as_of_date": r.d, **b, "prompt_version": "v1", "facts_json": r.facts_json})
    return pd.DataFrame(rows), {**why, "llm_error": int(out.err.notna().sum())}


def main(cm):
    import score as sc
    from pyspark.sql import SparkSession
    c, sp = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate()
    abac, league, club = extract(sp, c)
    b, why = briefs(sp, c, cm, sc)
    P, B, IL = (c.fq("armguard_ops", t) for t in ("pitcher_risk_current", "pitcher_briefs", "il_extractions"))
    sc.sdf(sp, b, BRIEF_OUT).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(B)
    sp.sql(f"MERGE INTO {P} t USING (SELECT * FROM {B} WHERE as_of_date = DATE(:d)) s ON t.athlete_id = s.athlete_id "
           "WHEN MATCHED THEN UPDATE SET t.headline = s.headline, t.recommended_action = s.recommended_action "
           "WHEN NOT MATCHED BY SOURCE THEN UPDATE SET t.headline = NULL, t.recommended_action = NULL", args={"d": c.as_of_date})
    k = sp.sql(f"SELECT coalesce(body_part, 'unspecified') body_part, count(*) placements FROM {IL} WHERE lower(position) RLIKE '^(rhp|lhp)$|pitcher' "
               "GROUP BY 1 ORDER BY 2 DESC, 1").toPandas()
    h = b[(b.athlete_id == cm.HERO) & (b.as_of_date.astype(str) == c.as_of_date)][["headline", "recommended_action", "fallback"]]
    cm.write_evidence(c, "t7_genai", "GenAI: IL extraction KPIs and pitcher briefs",
        f"Real 2025 MLB pitcher IL placements by body part ({sp.table(IL).count()} league placements parsed):\n"
        f"{k.head(6).to_markdown(index=False)}\nTotal pitcher IL placements: {k.placements.sum()}\n"
        f"New/pending this run: league {league}, club {club} (medical; {'ABAC' if abac else 'SET MASK'})\n\n"
        f"Briefs: {len(b)} over {b.as_of_date.nunique()} dates; fallback {b.fallback.sum()} ({b.fallback.mean():.0%}); reasons {why}\n"
        f"Hero {cm.HERO} brief {c.as_of_date}:\n{h.to_markdown(index=False)}")


if __name__ == "__main__":
    R = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path[:0] = [R, R + "/3_ml"]
    import common; common.bootstrap(); main(common)
