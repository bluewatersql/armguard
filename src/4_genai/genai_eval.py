"""t10: GenAI eval (spec §13 A5): ai_extract body-part accuracy, ai_classify triage F1, brief scorers in MLflow GenAI."""
import datetime as dt, json, os, re, sys
import pandas as pd

LABELS, EXP, PV = ["none", "fatigue", "soreness", "red_flag"], "/Shared/armguard-eval", "v1"
EV = ("ts TIMESTAMP, prompt_version STRING, n_briefs INT, extract_n INT, extract_correct INT, extract_acc DOUBLE, triage_n INT, "
      "triage_macro_f1 DOUBLE, grounded DOUBLE, no_leak DOUBLE, playbook DOUBLE, judge DOUBLE, notes_readable BOOLEAN, mode STRING, "
      "mlflow_run_id STRING")
_W = re.compile(r"[a-z0-9']+")


def grams(text, n=6):
    w = _W.findall((text or "").lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def leaks(response, notes, n=6):
    """True if any n-word window of any note appears verbatim (case/punctuation-insensitive) in the response."""
    r = grams(response, n)
    return any(r & grams(t, n) for t in notes)


def checks(cm, rec, notes):
    """The three scorer values for one eval record; the MLflow scorers call the same expressions."""
    o = rec["outputs"]["response"]
    return {"brief_grounded": cm.numbers_grounded(o, json.loads(rec["inputs"]["facts"])),
            "no_medical_leak": not leaks(o, notes.get(rec["inputs"]["athlete_id"], ())),
            "action_in_playbook": rec["expectations"]["action"] in cm.PLAYBOOK}


def mlflow_eval(cm, c, data, notes, rates, judge):
    """Log to /Shared/armguard-eval (tag prompt_version). Scorer values only: no note text in traces, metrics or artifacts."""
    import mlflow
    from mlflow.genai.scorers import scorer

    @scorer
    def brief_grounded(inputs, outputs):
        return cm.numbers_grounded(outputs["response"], json.loads(inputs["facts"]))

    @scorer
    def no_medical_leak(inputs, outputs):
        return not leaks(outputs["response"], notes.get(inputs["athlete_id"], ()))

    @scorer
    def action_in_playbook(expectations):
        return expectations["action"] in cm.PLAYBOOK

    sc = [brief_grounded, no_medical_leak, action_in_playbook]
    if judge:
        from mlflow.genai.scorers import Guidelines
        if not re.fullmatch(r"databricks-[a-z0-9-]+", c.judge_endpoint):
            raise SystemExit(f"invalid judge_endpoint {c.judge_endpoint!r}")
        sc.append(Guidelines(name="coach_tone", model=f"databricks:/{c.judge_endpoint}", guidelines=(
            "The response must be a concise, actionable arm-care note for a pitching coach and must not state a medical diagnosis.")))
    mlflow.set_experiment(EXP)
    with mlflow.start_run(run_name=f"genai_eval_{PV}", tags={"prompt_version": PV}) as run:
        try:
            met, mode = mlflow.genai.evaluate(data=data, scorers=sc).metrics, "mlflow.genai.evaluate"
        except Exception as e:  # serverless v3 incompatibility -> same scorers in plain Python, metrics still logged
            print("genai.evaluate failed:", type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "")
            met, mode = {}, f"plain-python fallback ({type(e).__name__})"
        mlflow.log_metrics({f"py_{k}": v for k, v in rates.items() if v is not None})
    jm = next((v for k, v in met.items() if k.startswith("coach_tone") and k.endswith("/mean")), None)
    agree = {k: met.get(f"{k}/mean") for k in ("brief_grounded", "no_medical_leak", "action_in_playbook")}
    return run.info.run_id, mode, jm, agree


def main(cm):
    from pyspark.sql import SparkSession
    from sklearn.metrics import confusion_matrix, f1_score
    c, sp = cm.Cfg.from_args(sys.argv[1:]), SparkSession.builder.getOrCreate()
    judge = any(a == "--judge=1" for a in sys.argv[1:])
    A, O = (lambda t: c.fq("armguard", t)), (lambda t: c.fq("armguard_ops", t))
    x = sp.sql(f"""SELECT count(*) n, count(e.event_id) extracted, count_if(lower(e.body_part) = lower(i.body_part)) ok
      FROM {c.fq('armguard_gt', 'incidents')} i JOIN {A('silver_il_events')} s
        ON s.athlete_id = i.athlete_id AND s.event_date = i.il_date AND s.action = 'placed'
      LEFT JOIN {A('club_il_extractions')} e ON e.event_id = s.event_id WHERE i.kind = 'injury'""").first()
    acc = x.ok / x.n if x.n else None
    tri = sp.sql(f"SELECT l.true_category y, t.note_flag p FROM {O('note_triage')} t JOIN {c.fq('armguard_gt', 'note_labels')} l "
                 "USING (note_id)").toPandas()
    cmx = pd.DataFrame(confusion_matrix(tri.y, tri.p, labels=LABELS), index=[f"true {k}" for k in LABELS], columns=LABELS)
    f1 = float(f1_score(tri.y, tri.p, labels=LABELS, average="macro", zero_division=0))
    b = sp.sql(f"SELECT athlete_id, as_of_date, headline, why, recommended_action, facts_json FROM {O('pitcher_briefs')} "
               "WHERE prompt_version = :v ORDER BY as_of_date, athlete_id", args={"v": PV}).toPandas()
    notes = {}
    for a, t in sp.sql(f"SELECT athlete_id, note_text FROM {A('silver_trainer_notes')}").collect():
        notes.setdefault(a, []).append(t or "")
    readable = bool(notes) and not any(t.startswith("[REDACTED") for v in notes.values() for t in v)
    data = [{"inputs": {"facts": r.facts_json, "athlete_id": r.athlete_id}, "outputs": {"response": f"{r.headline} {r.why}"},
             "expectations": {"action": r.recommended_action}} for r in b.itertuples()]
    v = pd.DataFrame([checks(cm, d, notes) for d in data], columns=["brief_grounded", "no_medical_leak", "action_in_playbook"])
    rt = {k: float(v[k].mean()) if len(v) else None for k in v.columns}
    run_id, mode, jm, agree = mlflow_eval(cm, c, data, notes, {**rt, "extract_acc": acc, "triage_macro_f1": f1}, judge)
    T = O("eval_genai")
    sp.createDataFrame([(dt.datetime.now(dt.timezone.utc), PV, len(b), x.n, x.ok, acc, len(tri), f1, rt["brief_grounded"],
                         rt["no_medical_leak"], rt["action_in_playbook"], jm, readable, mode, run_id)], EV).write.mode("append").saveAsTable(T)
    hist = sp.sql(f"SELECT date_format(ts, 'yyyy-MM-dd HH:mm') ts, n_briefs, round(extract_acc, 3) extract_acc, round(triage_macro_f1, 3) f1, "
                  f"round(grounded, 3) grounded, round(no_leak, 3) no_leak, round(playbook, 3) playbook, mode FROM {T} ORDER BY ts DESC LIMIT 5").toPandas()
    pf = lambda ok: "PASS" if ok else "MISS"
    g, nl, pb = (rt[k] or 0 for k in v.columns)
    cm.write_evidence(c, "t10_genai_eval", "GenAI evaluation (spec §13 A5)",
        f"ai_extract body part vs ground truth: {x.ok}/{x.n} = {acc or 0:.1%} ({x.extracted} extracted) -> {pf((acc or 0) >= .85)} (>= 85%)\n"
        f"ai_classify triage: {len(tri)} notes, macro-F1 {f1:.3f} (reported)\n\n{cmx.to_markdown()}\n\n"
        f"Briefs (prompt_version {PV}): {len(b)} briefs (target 50, R30); response = headline + why\n"
        f"- brief_grounded {g:.1%} -> {pf(g >= .9)} (>= 90%)\n- no_medical_leak {nl:.1%} -> {pf(nl == 1)} (= 100%; 6-word note windows; "
        f"notes readable by job identity: {'yes' if readable else 'NO - check vacuous'})\n- action_in_playbook {pb:.1%} -> {pf(pb == 1)} (= 100%)\n"
        f"- LLM judge coach_tone: {'off (enable with --judge=1)' if not judge else jm}\n\n"
        f"MLflow: experiment {EXP}, run {run_id}, tag prompt_version={PV}, mode {mode}\nMLflow scorer means: {agree}\n\n"
        f"Eval history (last 5):\n{hist.to_markdown(index=False)}", limit=5000)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    import common; common.bootstrap(); main(common)
