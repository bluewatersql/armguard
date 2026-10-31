"""t10: the holdout evaluation of the frozen @champion vs the staff rule (spec §13 A4). Gated (job param validate=1),
once per frozen configuration (config hash) unless --force=1. Metrics live in holdout.py."""
import datetime as dt, os, sys
import pandas as pd

HSTART, HERO_DATE, D = "2025-06-29", "2025-08-15", pd.Timedelta("1D")  # HSTART = earliest holdout IL (07-20) - 21 days
EV = ("ts TIMESTAMP, model_version STRING, high DOUBLE, model_det INT, rule_det INT, n_incidents INT, model_median_lead DOUBLE, "
      "rule_median_lead DOUBLE, model_hn_fa INT, rule_hn_fa INT, model_null DOUBLE, rule_null DOUBLE, null_pairs INT, hero_p1 BOOLEAN, "
      "a4_pass BOOLEAN, holdout_evaluations INT, config_hash STRING")
H = {"key": "incident_id", "il_date": "date", "m_det": "model detected", "m_lead": "model lead", "r_det": "rule detected",
     "r_lead": "rule lead", "m_run": "model run", "m_p1": "model P1", "m_fa": "model FA", "r_fa": "rule FA"}


def main(cm):
    import holdout as ho, mlflow, score as sc, train as tr
    from pyspark.sql import SparkSession
    c, sp = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate()
    force = "--force=1" in sys.argv[1:]
    t = sp.table(c.fq("armguard_ops", "model_thresholds")).first()
    g = sp.table(c.fq("armguard_gt", "generator_manifest")).first()
    h = ho.config_hash(tr.FEATURES, tr.PARAMS, t.high, t.elevated, g.seed, g.version)
    T = c.fq("armguard_ops", "eval_validation")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {T} ({EV})")
    if "config_hash" not in sp.table(T).columns:
        sp.sql(f"ALTER TABLE {T} ADD COLUMNS (config_hash STRING)")
    seen = sp.sql(f"SELECT count(*) FROM {T} WHERE config_hash = :h", args={"h": h}).first()[0]
    if seen and not force:  # evidence of the earlier evaluation stays untouched
        return print(f"SKIP: already evaluated (config {h}, {seen} row(s)); pass --force=1 to re-evaluate")
    nm = f"{c.catalog}.armguard_ops.injury_risk"
    mlflow.set_registry_uri("databricks-uc")
    v = mlflow.MlflowClient().get_model_version_by_alias(nm, "champion").version
    if str(v) != str(t.model_version):
        raise SystemExit(f"config error: model_thresholds is for v{t.model_version} but @champion is v{v}")
    memo = {}
    day = lambda d: memo[d] if d in memo else memo.setdefault(d, sc.score_date(sp, c, d))
    hero = day(HERO_DATE)  # also loads @champion into sc._MODEL
    hero = hero[hero.athlete_id == cm.HERO]
    rows = tr.load_rows(sp, c).query(f"feature_date >= '{HSTART}'")
    rows = rows.assign(risk_score=sc._MODEL["m"].predict(rows[tr.FEATURES]).risk_score.values)
    inc = sp.sql(f"SELECT incident_id key, athlete_id, kind, sub_kind, il_date, split FROM {c.fq('armguard_gt', 'incidents')} "
                 "ORDER BY il_date, incident_id").toPandas()
    inc["il_date"] = pd.to_datetime(inc.il_date)
    never, inc = set(inc.athlete_id), inc[inc.split == "holdout"].drop(columns="split")
    p1d = lambda a, d: bool(((d.athlete_id == a) & (d.priority == "P1")).any())
    p1 = {k: any(p1d(a, day(f"{e + i * D:%Y-%m-%d}")) for i in range(8))
          for k, a, e in inc[inc.sub_kind == "dead_arm_note"][["key", "athlete_id", "il_date"]].itertuples(index=False)}
    per, neg, S = ho.evaluate(rows, inc, t.high, p1, never)
    hp1 = bool(hero.priority.eq("P1").any())
    chk = ho.a4(S, len(per), hp1, HERO_DATE)
    ok = all(x for _, x, _ in chk if x is not None)
    n = sp.sql(f"SELECT coalesce(max(holdout_evaluations), 0) FROM {T}").first()[0] + 1
    nan = lambda x: None if pd.isna(x) else float(x)
    m, r = S["m"], S["r"]
    sp.createDataFrame([(dt.datetime.now(dt.timezone.utc), t.model_version, float(t.high), m["det"], r["det"], len(per), nan(m["lead"]),
                         nan(r["lead"]), m["fa"], r["fa"], nan(m["null"]), nan(r["null"]), m["pairs"], hp1, ok, n, h)], EV) \
        .write.mode("append").saveAsTable(T)
    show = lambda d, cols: d[cols].astype(object).apply(lambda s: s.map(ho.cell)).rename(columns=H).to_markdown(index=False)
    md = lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    hl = "; ".join(f"{x.tier} {x.priority}, risk {x.risk_score:.3f}, note {x.note_flag}" for x in hero.itertuples()) or "not scored"
    nr = lambda k, p="null": f"{S[k][p]:.1%} ({S[k][p + '_det']}/{S[k]['pairs' if p == 'null' else p + '_pairs']})"
    cm.write_evidence(c, "t10_validation", f"Holdout validation #{n} (spec §13 A4)",
        f"@champion v{t.model_version}, model alert risk_score >= HIGH {t.high}; staff rule velo_delta <= -1.5; config {h}\n"
        f"{len(rows)} holdout pitcher-days scored ({HSTART}..), training filters; detection = alert run of >= 2 consecutive days "
        "within [IL-21, IL-1] (days inside the window only); lead = IL - first in-window alert day (max 21)\n\n"
        f"{show(per, ['key', 'athlete_id', 'il_date', 'm_det', 'm_lead', 'r_det', 'r_lead'])}\n\n"
        "Hard negatives (false alarm = >= 2 alert days within [event-3, event+10]; dead_arm_note: model P1 on event..event+7):\n"
        f"{show(neg, ['key', 'athlete_id', 'sub_kind', 'il_date', 'm_run', 'm_p1', 'm_fa', 'r_fa'])}\n\n"
        "| metric | model | staff rule |\n|---|---|---|\n"
        f"| recall | {m['det']}/{len(per)} | {r['det']}/{len(per)} |\n| median lead (detected) | {md(m['lead'])} | {md(r['lead'])} |\n"
        f"| hard-negative false alarms | {m['fa']}/{len(neg)} | {r['fa']}/{len(neg)} |\n"
        f"| null detection rate | {nr('m')} | {nr('r')} |\n| null, never-injured only | {nr('m', 'strict')} | {nr('r', 'strict')} |\n\n"
        f"Null set: {m['athletes']} pitchers with no holdout incident ({m['strict_athletes']} never in any incident) x "
        f"{per.il_date.nunique()} holdout IL dates; a pair needs >= 2 eligible days in its window. The windows overlap in time, "
        "so pairs are not independent.\n\n"
        + "\n".join(f"- {k}: {'REPORTED' if x is None else 'PASS' if x else 'MISS'} {d}" for k, x, d in chk)
        + f"\n\nHero {cm.HERO} on {HERO_DATE}: {hl}\nA4 overall: {'PASS' if ok else 'MISS'}\nholdout_evaluations: {n}", limit=6000)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap(); main(common)
