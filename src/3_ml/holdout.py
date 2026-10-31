"""Pure holdout metrics for t10 (spec §13 A4): in-window alert runs, lead, hard negatives, null rate, config hash."""
import hashlib, json
import numpy as np, pandas as pd

D = pd.Timedelta("1D")


def windows(w, lo, hi):
    return w.assign(lo=w.il_date + lo * D, hi=w.il_date + hi * D)


def first_hit(runs, w):
    """First in-window alert day per window [lo, hi]; a run counts only with >= 2 consecutive days inside the window (NaT = none)."""
    h = w.merge(runs, on="athlete_id")
    if h.empty:
        return pd.Series(pd.NaT, index=w.index, dtype="datetime64[ns]")
    h = h.assign(s=h.run_start.where(h.run_start > h.lo, h.lo), e=h.run_end.where(h.run_end < h.hi, h.hi))
    h = h[h.e - h.s >= D]
    return w.key.map(h.groupby("key").s.min()) if len(h) else pd.Series(pd.NaT, index=w.index, dtype="datetime64[ns]")


def detect(runs, il):
    """Detected = an alert run of >= 2 consecutive days within [IL-21, IL-1]; lead = IL - first in-window alert day (max 21)."""
    w = windows(il, -21, -1)
    s = first_hit(runs, w)
    return pd.DataFrame({"det": s.notna(), "lead": (w.il_date - s).dt.days}, index=w.index)


def null_pairs(rows, athletes, dates, min_days=2):
    """Pseudo-IL windows: athlete x holdout IL date, kept if the athlete has >= min_days eligible days in [d-21, d-1]."""
    p = pd.DataFrame([(f"{a}@{d:%m-%d}", a, d) for a in athletes for d in dates], columns=["key", "athlete_id", "il_date"])
    w = windows(p, -21, -1).merge(rows[["athlete_id", "feature_date"]], on="athlete_id")
    n = w[(w.feature_date >= w.lo) & (w.feature_date <= w.hi)].key.value_counts()
    return p[p.key.isin(n[n >= min_days].index)]


def evaluate(rows, inc, high, p1, never):
    """rows: scored pitcher-days; inc: holdout incidents (key, athlete_id, kind, sub_kind, il_date); p1: {key: P1 on
    event..event+7} for dead_arm_note; never: athletes with an incident in any split (strict null sensitivity).
    -> (per-incident, hard negatives, summary per m=model / r=staff rule)."""
    import common as cm, train as tr
    rows = rows.assign(m=rows.risk_score >= high, r=rows.velo_delta.map(cm.baseline_flag).astype(bool))
    il, hn = inc[inc.kind == "injury"], inc[inc.kind == "hard_negative"]
    free = sorted(set(rows.athlete_id) - set(inc.athlete_id))
    dates = sorted(set(il.il_date))
    pairs, strict = null_pairs(rows, free, dates), null_pairs(rows, [a for a in free if a not in never], dates)
    per, neg, S = il[["key", "athlete_id", "il_date"]].copy(), hn[["key", "athlete_id", "sub_kind", "il_date"]].copy(), {}
    rate = lambda x: x.mean() if len(x) else float("nan")
    for k in "mr":
        runs = tr.alert_runs(rows, k)
        d, nd, sd = detect(runs, il), detect(runs, pairs).det, detect(runs, strict).det
        per[f"{k}_det"], per[f"{k}_lead"] = d.det, d.lead
        neg[f"{k}_run"] = first_hit(runs, windows(hn, -3, 10)).notna()
        S[k] = {"det": int(d.det.sum()), "lead": d.lead[d.det].median(), "null": rate(nd), "null_det": int(nd.sum()),
                "pairs": len(pairs), "athletes": pairs.athlete_id.nunique(), "strict": rate(sd), "strict_det": int(sd.sum()),
                "strict_pairs": len(strict), "strict_athletes": strict.athlete_id.nunique()}
    neg["m_p1"] = [p1.get(k) if s == "dead_arm_note" else None for k, s in zip(neg.key, neg.sub_kind)]
    neg["m_fa"] = [bool(p) if s == "dead_arm_note" else bool(r) for s, p, r in zip(neg.sub_kind, neg.m_p1, neg.m_run)]
    neg["r_fa"] = neg.r_run
    for k in "mr":
        S[k].update(fa=int(neg[f"{k}_fa"].sum()), hn=len(neg))
    return per, neg, S


def a4(S, n, hero_p1, hero_date="2025-08-15"):
    """[(criterion, ok, detail)]; ok None = reported only. FA ties are broken by the null rate (fixed before the holdout run)."""
    m, r = S["m"], S["r"]
    return [("recall >= 6/8", m["det"] * 8 >= 6 * n, f"{m['det']}/{n}"),
            ("median lead >= 7 d", bool(m["lead"] >= 7), f"{m['lead']:.1f} d"),
            ("<= 1 hard-negative false alarm", m["fa"] <= 1, f"{m['fa']}/{m['hn']}"),
            ("null detection rate reported", None, f"model {m['null']:.0%}, rule {r['null']:.0%} ({m['pairs']} pairs, {m['athletes']} pitchers)"),
            ("model beats staff rule on false alarms at >= recall", m["det"] >= r["det"] and (m["fa"], m["null"]) < (r["fa"], r["null"]),
             f"FA {m['fa']} vs {r['fa']}, null {m['null']:.0%} vs {r['null']:.0%}, recall {m['det']} vs {r['det']}"),
            (f"hero P1 on {hero_date}", bool(hero_p1), "")]


def cell(v):
    if v is None or isinstance(v, float) and np.isnan(v):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "Y" if v else "N"
    return f"{v:%Y-%m-%d}" if isinstance(v, pd.Timestamp) else f"{v:g}" if isinstance(v, float) else str(v)


def config_hash(features, params, high, elevated, seed, version):
    """Frozen-configuration id (not the model version): features, hyperparameters, thresholds, simulator seed/version."""
    cfg = {"features": list(features), "params": params, "high": round(float(high), 6), "elevated": round(float(elevated), 6),
           "seed": int(seed), "version": str(version)}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
