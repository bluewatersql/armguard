import os, sys
import pandas as pd

FEATURES = ["velo_delta", "spin_delta", "rel_z_delta", "ext_delta", "acwr", "pitches_7d", "days_since_app",
    "soreness_3d", "min_temp_7d", "role_sp", "appearances_30d"]
END, D = "2025-06-30", pd.Timedelta("1D")
def load_rows(sp, c):
    d = sp.table(c.fq("armguard", "gold_pitcher_daily_features")).filter(
        "NOT on_il AND n_prior_apps >= 5 AND days_since_app <= 30").toPandas()
    return d.astype(dict.fromkeys(FEATURES, float)).assign(feature_date=pd.to_datetime(d.feature_date), label=d.label_il_14d)
def alert_runs(df, flag):
    d = df[df[flag]].sort_values(["athlete_id", "feature_date"])
    b = (d.feature_date.diff().dt.days != 1) | (d.athlete_id != d.athlete_id.shift())
    g = d.groupby(["athlete_id", b.cumsum()]).feature_date.agg(run_start="min", run_end="max", n="count")
    return g[g.n >= 2].reset_index()[["athlete_id", "run_start", "run_end"]]
def incident_recall(r, il):
    h = il.merge(r, on="athlete_id")
    h = h[(h.run_start <= h.il_date - D) & (h.run_end >= h.il_date - 21 * D)]
    return h.athlete_id.nunique() / max(len(il), 1)
def false_alarm_rate(runs, il, n_nonlabel_days):
    h = runs.merge(il, on="athlete_id")
    h = h[(h.run_start <= h.il_date - D) & (h.run_end >= h.il_date - 21 * D)]
    return (len(runs) - len(h.drop_duplicates(["athlete_id", "run_start"]))) * 100 / n_nonlabel_days
def ev(d, f, il):
    d = d.assign(f=f); r = alert_runs(d, "f")
    return incident_recall(r, il), false_alarm_rate(r, il, (d.label == 0).sum())
def main(cm):
    import mlflow, pyspark.sql as P, sklearn.ensemble as se, sklearn.model_selection as ms
    c, sp = cm.Cfg.from_args(sys.argv[1:]), P.SparkSession.builder.getOrCreate()
    d = load_rows(sp, c).query(f"feature_date <= '{END}'")
    il = sp.sql(f"SELECT athlete_id, TIMESTAMP(event_date) il_date FROM {c.fq('armguard', 'silver_il_events')} "
        f"WHERE action = 'placed' AND event_date <= '{END}'").toPandas()
    m, X, y = se.HistGradientBoostingClassifier(max_depth=3, max_iter=100, min_samples_leaf=50, class_weight="balanced"), d[FEATURES], d.label
    oof = ms.cross_val_predict(m, X, y, groups=d.athlete_id, cv=ms.GroupKFold(4), method="predict_proba")[:, 1]
    t, rc, fa = min(((t, *ev(d, oof >= t, il)) for t in [i / 100 for i in range(5, 96)]), key=lambda s: (max(s[2], 3), -s[1] * (s[2] <= 3)))
    el = round(0.6 * t, 3)
    sr, sf = ev(d, d.velo_delta.map(cm.baseline_flag), il)
    m.fit(X, y)
    class InjuryRisk(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return pd.DataFrame({"risk_score": m.predict_proba(model_input[FEATURES])[:, 1]})
    w, x, nm = InjuryRisk(), X.head(5), f"{c.catalog}.armguard_ops.injury_risk"
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Shared/armguard-model")
    with mlflow.start_run() as r:
        i = mlflow.pyfunc.log_model(name="model", python_model=w, registered_model_name=nm, input_example=x,
            signature=mlflow.models.infer_signature(x, w.predict(None, x)), pip_requirements=["scikit-learn==1.6.1", "pandas", "numpy"])
    v = i.registered_model_version
    mlflow.MlflowClient().set_registered_model_alias(nm, "champion", v)
    o = mlflow.pyfunc.load_model(f"models:/{nm}@champion").predict(x)
    assert list(o.columns) == ["risk_score"] and o.risk_score.between(0, 1).all()
    sp.sql(f"CREATE OR REPLACE TABLE {c.fq('armguard_ops', 'model_thresholds')} AS SELECT {t}D high, {el}D elevated, "
        f"'{v}' model_version, {rc}D oof_incident_recall, {fa}D oof_fa_per100")
    cm.write_evidence(c, "t5_model", "Model",
        f"rows {len(d)}, positives {y.sum()}\nHIGH {t}, ELEVATED {el}\nmodel recall {rc:.2f}, fa/100 {fa:.2f}\n"
        f"staff recall {sr:.2f}, fa/100 {sf:.2f}\n{nm} v{v}, run {r.info.run_id}\nround-trip load OK")
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap(); main(common)
