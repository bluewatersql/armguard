"""t6: ai_classify note triage, @champion scoring, priority queue -> pitcher_risk_current (CDF)."""
import os, sys
import pandas as pd

SEV_SQL = "array('none', 'fatigue', 'soreness', 'red_flag')"  # least -> most severe
LABELS_VERSION = "v1"  # bump on any LABELS change: t6 drops and re-classifies note_triage rows of other versions
LABELS = ('{"none":"Routine, no concern","fatigue":"Tiredness, heavy arm, slow recovery","soreness":"Localized soreness or tightness",'
          '"red_flag":"Pain on release, numbness, grip loss, imaging advised"}')
# v2 (not shipped: triage macro-F1 0.878 < v1 0.891 on fevm; soreness -> none): {"none":"Routine note, no arm concern",
# "fatigue":"Tiredness, heavy or dead arm, slow recovery; no pain","soreness":"Mild or dull soreness or tightness; still throwing",
# "red_flag":"Sharp or acute pain, pain on release, numbness or tingling, grip loss, shut down from throwing, or imaging/physician referral"}
Z = ["velo_delta", "soreness_3d", "acwr", "spin_delta", "pitches_7d"]
DDL = ("athlete_id STRING NOT NULL, display_name STRING, as_of_date DATE, risk_score DOUBLE, tier STRING, note_flag STRING, priority STRING, "
       "top_drivers STRING, baseline_flag BOOLEAN, velo_delta DOUBLE, pitches_7d DOUBLE, acwr DOUBLE, headline STRING, recommended_action STRING, "
       "model_version STRING")
COLS = [col.split()[0] for col in DDL.split(", ")]
KEEP = ("headline", "recommended_action")  # owned by t7: a re-score keeps the last good brief
_MODEL = {}


def sdf(sp, df, ddl):
    """pandas -> Spark with an explicit schema (NaN/None -> NULL)."""
    return sp.createDataFrame(df[[col.split()[0] for col in ddl.split(", ")]], ddl)


def top_drivers(h, a):
    """For rows on date a: the 3 largest |z| of Z vs the athlete's own season-to-date mean/std, as 'feature value; ...'."""
    g = h.groupby("athlete_id")[Z]
    z = ((h[Z] - g.transform("mean")) / g.transform("std")).abs()
    r = h[h.feature_date == a]
    return pd.Series(["; ".join(f"{k} {round(r.at[i, k], 1):g}" for k in z.loc[i].dropna().nlargest(3).index) for i in r.index],
                     index=r.index, dtype=object)


def triage(sp, c):
    """Incremental and version-aware: rows of another LABELS_VERSION (or NULL = pre-versioning) are deleted, then every note
    not in note_triage is classified; failed calls stay out and are retried next run. -> (table, cache summary)."""
    o, v = c.fq("armguard_ops", "note_triage"), {"v": LABELS_VERSION}
    sp.sql(f"CREATE TABLE IF NOT EXISTS {o} (note_id STRING, athlete_id STRING, note_ts TIMESTAMP, note_flag STRING, confidence DOUBLE, "
           "labels_version STRING)")
    if "labels_version" not in sp.table(o).columns:
        sp.sql(f"ALTER TABLE {o} ADD COLUMNS (labels_version STRING)")
    before = dict(sp.sql(f"SELECT coalesce(labels_version, 'v1 (unversioned)'), count(*) FROM {o} GROUP BY 1").collect())
    stale = {k: n for k, n in before.items() if k != LABELS_VERSION}
    if stale:
        sp.sql(f"DELETE FROM {o} WHERE labels_version IS NULL OR labels_version <> :v", args=v)
    sp.sql(f"""INSERT INTO {o} (note_id, athlete_id, note_ts, note_flag, confidence, labels_version)
      SELECT note_id, athlete_id, note_ts, c:response[0]:value::string, c:response[0]:confidence_score::double, :v FROM (
        SELECT n.*, ai_classify(n.note_text, :labels, MAP('version','2.1','enableConfidenceScores','true')) c
        FROM {c.fq('armguard', 'silver_trainer_notes')} n LEFT ANTI JOIN {o} t USING (note_id))
      WHERE c:response[0]:value IS NOT NULL""", args={**v, "labels": LABELS})
    now = sp.sql(f"SELECT count(*) FROM {o} WHERE labels_version = :v", args=v).first()[0]
    notes = sp.table(c.fq("armguard", "silver_trainer_notes")).count()
    new = now - before.get(LABELS_VERSION, 0)
    msg = (f"triage cache: before {before}; re-classified {sum(stale.values())} notes (labels {', '.join(sorted(stale))} -> "
           f"{LABELS_VERSION}); " if stale else f"triage cache: labels {LABELS_VERSION}, no stale rows; ")
    return o, msg + f"{new} classified this run; {now}/{notes} notes triaged ({notes - now} failed, retried next run)"


def score_date(sp, c, as_of):
    """Score every eligible pitcher on as_of with @champion; add note_flag, tier, priority, baseline_flag, top_drivers."""
    import common as cm, mlflow, train as tr
    if not _MODEL:
        mlflow.set_registry_uri("databricks-uc")
        _MODEL["m"] = mlflow.pyfunc.load_model(f"models:/{c.catalog}.armguard_ops.injury_risk@champion")
    a, t = pd.Timestamp(as_of), sp.table(c.fq("armguard_ops", "model_thresholds")).first()
    h = tr.load_rows(sp, c).query("feature_date <= @a")  # season to date: no look-ahead
    r = h[h.feature_date == a].copy()
    if r.empty:  # no eligible pitcher that day (e.g. off-season date)
        return pd.DataFrame(columns=list(dict.fromkeys([*COLS, *tr.FEATURES])))
    r["risk_score"] = _MODEL["m"].predict(r[tr.FEATURES]).risk_score.values
    r["top_drivers"] = top_drivers(h, a)
    flags = dict(sp.sql(f"SELECT athlete_id, max_by(note_flag, array_position({SEV_SQL}, note_flag)) FROM {c.fq('armguard_ops', 'note_triage')} "
                        "WHERE to_date(note_ts) BETWEEN date_sub(DATE(:d), 7) AND DATE(:d) GROUP BY 1", args={"d": as_of}).collect())
    r["note_flag"] = [flags.get(x) for x in r.athlete_id]
    r["tier"] = [cm.tier(s, t.high, t.elevated) for s in r.risk_score]
    r["priority"] = [cm.priority(x, y) for x, y in zip(r.tier, r.note_flag)]
    r["baseline_flag"] = [cm.baseline_flag(None if pd.isna(v) else v) for v in r.velo_delta]
    return r.assign(as_of_date=a.date(), model_version=t.model_version, headline=None, recommended_action=None)


def main(cm):
    from pyspark.sql import SparkSession
    c, sp = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate()
    (o, cache), T = triage(sp, c), c.fq("armguard_ops", "pitcher_risk_current")
    r = score_date(sp, c, c.as_of_date)
    if r.empty:  # never MERGE an empty source: NOT MATCHED BY SOURCE would empty the queue
        raise SystemExit(f"no eligible pitchers on {c.as_of_date}")
    sdf(sp, r, DDL).createOrReplaceTempView("risk_src")
    sp.sql(f"CREATE TABLE IF NOT EXISTS {T} ({DDL}) TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    upd = ", ".join(f"t.{k} = s.{k}" for k in COLS if k not in KEEP)
    sp.sql(f"MERGE INTO {T} t USING risk_src s ON t.athlete_id = s.athlete_id WHEN MATCHED THEN UPDATE SET {upd} "
           "WHEN NOT MATCHED THEN INSERT * WHEN NOT MATCHED BY SOURCE THEN DELETE")
    try:  # informational RELY PK; already present on re-runs
        sp.sql(f"ALTER TABLE {T} ADD CONSTRAINT risk_pk PRIMARY KEY (athlete_id) RELY")
    except Exception as e:
        print("PK:", str(e).splitlines()[0][:120])
    K = "display_name priority tier risk_score note_flag top_drivers".split()
    q = r.dropna(subset=["priority"]).sort_values(["priority", "risk_score"], ascending=[True, False])[K]
    hero = r[r.athlete_id == cm.HERO][K]
    md = lambda d: d.to_markdown(index=False, floatfmt=".3f")
    cm.write_evidence(c, "t6_score", f"Scoring and triage, as of {c.as_of_date}",
        f"{len(r)} pitchers scored with @champion v{r.model_version.iloc[0]}\n\nTier x priority:\n"
        f"{pd.crosstab(r.tier, r.priority.fillna('none')).to_markdown()}\n\n"
        f"note_flag (most severe, last 7 days): {r.note_flag.fillna('no note').value_counts().to_dict()}\n"
        f"note_triage, all notes: {dict(sp.table(o).groupBy('note_flag').count().collect())}\n{cache}\n\n"
        f"Top queue:\n{md(q.head(8))}\n\nHero {cm.HERO} must be P1: {'PASS' if hero.priority.eq('P1').any() else 'MISS'}\n{md(hero)}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap(); main(common)
