"""t10: the holdout evaluation of the frozen @champion vs the staff rule (spec §13 A4): recall, lead, hard negatives, null rate, hero."""
import datetime as dt, os, sys
import numpy as np, pandas as pd

HSTART, HERO_DATE, D = "2025-06-29", "2025-08-15", pd.Timedelta("1D")  # HSTART = earliest holdout IL (07-20) - 21 days
EV = ("ts TIMESTAMP, model_version STRING, high DOUBLE, model_det INT, rule_det INT, n_incidents INT, model_median_lead DOUBLE, "
      "rule_median_lead DOUBLE, model_hn_fa INT, rule_hn_fa INT, model_null DOUBLE, rule_null DOUBLE, null_pairs INT, hero_p1 BOOLEAN, "
      "a4_pass BOOLEAN, holdout_evaluations INT")


def windows(w, lo, hi):
    return w.assign(lo=w.il_date + lo * D, hi=w.il_date + hi * D)


def first_hit(runs, w):
    """Earliest start of an alert run overlapping each window [lo, hi] (NaT = none), aligned to w."""
    h = w.merge(runs, on="athlete_id")
    h = h[(h.run_start <= h.hi) & (h.run_end >= h.lo)]
    return w.key.map(h.groupby("key").run_start.min()) if len(h) else pd.Series(pd.NaT, index=w.index)


def detect(runs, il):
    """Detected = a >= 2-day run overlaps [IL-21, IL-1]; lead = IL - start of the earliest such run, capped at 21 days."""
    w = windows(il, -21, -1)
    s = first_hit(runs, w)
    return pd.DataFrame({"det": s.notna(), "lead": (w.il_date - s.mask(s < w.lo, w.lo)).dt.days}, index=w.index)


def null_pairs(rows, athletes, dates):
    """Pseudo-IL windows: each athlete x each holdout IL date, kept if the athlete has an eligible day in [d-21, d-1]."""
    p = pd.DataFrame([(f"{a}@{d:%m-%d}", a, d) for a in athletes for d in dates], columns=["key", "athlete_id", "il_date"])
    w = windows(p, -21, -1).merge(rows[["athlete_id", "feature_date"]], on="athlete_id")
    return p[p.key.isin(w[(w.feature_date >= w.lo) & (w.feature_date <= w.hi)].key)]


def evaluate(rows, inc, high, p1):
    """Pure A4 core. rows: scored pitcher-days; inc: holdout incidents (key, athlete_id, kind, sub_kind, il_date);
    p1: {key: P1 seen on event..event+7} for dead_arm_note. -> (per-incident, hard negatives, summary per m=model / r=rule)."""
    import common as cm, train as tr
    rows = rows.assign(m=rows.risk_score >= high, r=rows.velo_delta.map(cm.baseline_flag).astype(bool))
    il, hn = inc[inc.kind == "injury"], inc[inc.kind == "hard_negative"]
    pairs = null_pairs(rows, sorted(set(rows.athlete_id) - set(inc.athlete_id)), sorted(set(il.il_date)))
    per, neg, S = il[["key", "athlete_id", "il_date"]].copy(), hn[["key", "athlete_id", "sub_kind", "il_date"]].copy(), {}
    for k in "mr":
        runs = tr.alert_runs(rows, k)
        d = detect(runs, il)
        per[f"{k}_det"], per[f"{k}_lead"] = d.det, d.lead
        neg[f"{k}_run"] = first_hit(runs, windows(hn, -3, 10)).notna()
        nd = detect(runs, pairs).det
        S[k] = {"det": int(d.det.sum()), "lead": d.lead[d.det].median(), "null": nd.mean() if len(nd) else float("nan"),
                "null_det": int(nd.sum()), "pairs": len(pairs), "athletes": pairs.athlete_id.nunique()}
    neg["m_p1"] = [p1.get(k) if s == "dead_arm_note" else None for k, s in zip(neg.key, neg.sub_kind)]
    neg["m_fa"] = [bool(p) if s == "dead_arm_note" else bool(r) for s, p, r in zip(neg.sub_kind, neg.m_p1, neg.m_run)]
    neg["r_fa"] = neg.r_run
    for k in "mr":
        S[k].update(fa=int(neg[f"{k}_fa"].sum()), hn=len(neg))
    return per, neg, S


def cell(v):
    if v is None or isinstance(v, float) and np.isnan(v):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "Y" if v else "N"
    return f"{v:%Y-%m-%d}" if isinstance(v, pd.Timestamp) else f"{v:g}" if isinstance(v, float) else str(v)


def a4(S, n, hero_p1):
    """[(criterion, ok, detail)]; ok None = reported only. FA ties are broken by the null rate (fixed before the holdout run)."""
    m, r = S["m"], S["r"]
    return [("recall >= 6/8", m["det"] * 8 >= 6 * n, f"{m['det']}/{n}"),
            ("median lead >= 7 d", bool(m["lead"] >= 7), f"{m['lead']:.1f} d"),
            ("<= 1 hard-negative false alarm", m["fa"] <= 1, f"{m['fa']}/{m['hn']}"),
            ("null detection rate reported", None, f"model {m['null']:.0%}, rule {r['null']:.0%} ({m['pairs']} pairs, {m['athletes']} pitchers)"),
            ("model beats staff rule on false alarms at >= recall", m["det"] >= r["det"] and (m["fa"], m["null"]) < (r["fa"], r["null"]),
             f"FA {m['fa']} vs {r['fa']}, null {m['null']:.0%} vs {r['null']:.0%}, recall {m['det']} vs {r['det']}"),
            (f"hero P1 on {HERO_DATE}", bool(hero_p1), "")]


def main(cm):
    import score as sc, train as tr
    from pyspark.sql import SparkSession
    c, sp = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate()
    t = sp.table(c.fq("armguard_ops", "model_thresholds")).first()
    memo = {}
    day = lambda d: memo[d] if d in memo else memo.setdefault(d, sc.score_date(sp, c, d))
    hero = day(HERO_DATE)  # also loads @champion into sc._MODEL
    hero = hero[hero.athlete_id == cm.HERO]
    rows = tr.load_rows(sp, c).query(f"feature_date >= '{HSTART}'")
    rows = rows.assign(risk_score=sc._MODEL["m"].predict(rows[tr.FEATURES]).risk_score.values)
    inc = sp.sql(f"SELECT incident_id key, athlete_id, kind, sub_kind, il_date FROM {c.fq('armguard_gt', 'incidents')} "
                 "WHERE split = 'holdout' ORDER BY il_date, incident_id").toPandas()
    inc["il_date"] = pd.to_datetime(inc.il_date)
    p1d = lambda a, d: bool(((d.athlete_id == a) & (d.priority == "P1")).any())
    p1 = {k: any(p1d(a, day(f"{e + i * D:%Y-%m-%d}")) for i in range(8))
          for k, a, e in inc[inc.sub_kind == "dead_arm_note"][["key", "athlete_id", "il_date"]].itertuples(index=False)}
    per, neg, S = evaluate(rows, inc, t.high, p1)
    hp1 = bool(hero.priority.eq("P1").any())
    chk = a4(S, len(per), hp1)
    ok = all(x for _, x, _ in chk if x is not None)
    T = c.fq("armguard_ops", "eval_validation")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {T} ({EV})")
    n = sp.sql(f"SELECT coalesce(max(holdout_evaluations), 0) FROM {T}").first()[0] + 1
    nan = lambda v: None if pd.isna(v) else float(v)
    m, r = S["m"], S["r"]
    sp.createDataFrame([(dt.datetime.now(dt.timezone.utc), t.model_version, float(t.high), m["det"], r["det"], len(per), nan(m["lead"]),
                         nan(r["lead"]), m["fa"], r["fa"], nan(m["null"]), nan(r["null"]), m["pairs"], hp1, ok, n)], EV) \
        .write.mode("append").saveAsTable(T)
    H = {"key": "incident_id", "il_date": "date", "m_det": "model detected", "m_lead": "model lead", "r_det": "rule detected",
         "r_lead": "rule lead", "m_run": "model run", "m_p1": "model P1", "m_fa": "model FA", "r_fa": "rule FA"}
    show = lambda d, cols: d[cols].astype(object).apply(lambda s: s.map(cell)).rename(columns=H).to_markdown(index=False)
    md = lambda v: f"{v:.1f}" if pd.notna(v) else "-"
    h = "; ".join(f"{x.tier} {x.priority}, risk {x.risk_score:.3f}, note {x.note_flag}" for x in hero.itertuples()) or "not scored"
    cm.write_evidence(c, "t10_validation", f"Holdout validation #{n} (spec §13 A4)",
        f"@champion v{t.model_version}, model alert risk_score >= HIGH {t.high}; staff rule velo_delta <= -1.5\n"
        f"{len(rows)} holdout pitcher-days scored ({HSTART}..), training filters; detection = alert run >= 2 consecutive days "
        "overlapping [IL-21, IL-1]; lead = IL - run start, capped at 21 d\n\n"
        f"{show(per, ['key', 'athlete_id', 'il_date', 'm_det', 'm_lead', 'r_det', 'r_lead'])}\n\n"
        "Hard negatives (false alarm = alert run overlapping [event-3, event+10]; dead_arm_note: model P1 on event..event+7):\n"
        f"{show(neg, ['key', 'athlete_id', 'sub_kind', 'il_date', 'm_run', 'm_p1', 'm_fa', 'r_fa'])}\n\n"
        "| metric | model | staff rule |\n|---|---|---|\n"
        f"| recall | {m['det']}/{len(per)} | {r['det']}/{len(per)} |\n| median lead (detected) | {md(m['lead'])} | {md(r['lead'])} |\n"
        f"| hard-negative false alarms | {m['fa']}/{len(neg)} | {r['fa']}/{len(neg)} |\n"
        f"| null detection rate | {m['null']:.1%} ({m['null_det']}/{m['pairs']}) | {r['null']:.1%} ({r['null_det']}/{r['pairs']}) |\n\n"
        f"Null set: {m['athletes']} pitchers with no holdout incident x {per.il_date.nunique()} holdout IL dates (pairs with an eligible day)\n\n"
        + "\n".join(f"- {k}: {'REPORTED' if x is None else 'PASS' if x else 'MISS'} {d}" for k, x, d in chk)
        + f"\n\nHero {cm.HERO} on {HERO_DATE}: {h}\nA4 overall: {'PASS' if ok else 'MISS'}\nholdout_evaluations: {n}", limit=6000)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap(); main(common)
