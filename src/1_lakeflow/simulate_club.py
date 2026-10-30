"""t2: simulate club systems (roster, wellness, notes, IL wire) from landed GUMBO; seal truth in armguard_gt."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
import common; common.bootstrap()
import glob, json, shutil
import numpy as np, pandas as pd
from faker import Faker
import sim_core as sc
from sim_core import D, T

LAB = "none fatigue soreness red_flag".split()
DEAD = "Says arm feels dead and heavy after bullpen; velocity and command fine; no pain."
TPL = {k: t.split("|") for k, t in zip(LAB, (
    "No complaints.|Loose {s} {p}.|Normal recovery.|On schedule.|Checked {s} {p}: fine.|Long toss fine.",
    "Tired after big week.|Heavy {s} {p}, no pain.|Low energy.|Travel fatigue.|Tired {s} {p}, no pain.|Run down.",
    "Mild {s} {p} soreness.|Tight {s} {p}.|Tender {s} {p}.|Sore {s} {p} on long toss.|Achy {s} {p}.|Stiff {s} {p}.",
    "Sharp {s} {p} pain; shut down.|Tingling from {s} {p}.|Felt a pop in {s} {p}.|Pain in {s} {p} every throw."
    "|Grip down, {s} {p} pain.|Imaging for {s} {p}."))}

def make_roster(mlbam_ids, rng, hands, real_names):
    f, used, rows = Faker("en_US"), set(real_names), []
    f.seed_instance(int(rng.integers(1e9)))
    for i, m in enumerate(sorted({int(x) for x in mlbam_ids}), 1):
        while (n := f"{f.first_name_male()} {f.last_name()}") in used: pass
        used.add(n); rows.append((f"A{i:02d}", m, n, hands.get(m, "R")))
    return pd.DataFrame(rows, columns="athlete_id mlbam_id display_name throws".split())

def make_wellness(roster, apps, incidents, rng):
    inj, out = incidents[incidents.kind == "injury"], []
    for r in roster.sort_values("athlete_id").itertuples():
        a = apps[apps.mlbam_id == r.mlbam_id].groupby("game_date").pitches.sum()
        if a.empty: continue
        d = pd.date_range(a.index.min(), a.index.max()); n = len(d)
        s = rng.poisson(1.5, n) + sum(np.round(sc._ramp(d, i.true_onset_date, i.il_date) * rng.uniform(2, 4))
                                      for i in inj[inj.mlbam_id == r.mlbam_id].itertuples())
        p3 = a.reindex(d, fill_value=0).rolling(3, 1).sum().to_numpy()
        out.append(pd.DataFrame({"athlete_id": r.athlete_id, "survey_date": d.strftime("%Y-%m-%d"),
            "arm_soreness_0_10": np.clip(s, 0, 10).astype(int), "fatigue_0_10": np.minimum(10, rng.poisson(1, n) + p3 // 40).astype(int),
            "sleep_hours": rng.normal(7.5, .6, n).round(1)}))
    return pd.concat(out, ignore_index=True)

def make_notes(roster, apps, incidents, rng):
    th, aid, rows = dict(zip(roster.mlbam_id, roster.throws)), dict(zip(roster.mlbam_id, roster.athlete_id)), []
    def note(m, day, lab, part=None, text=None):
        s = "right" if th[m] == "R" else "left"
        rows.append((T(day), aid[m], text or str(rng.choice(TPL[lab])).format(s=s, p=part or rng.choice(sc.BODY)), lab))
    for m, g in apps.groupby("mlbam_id"):  # ~8 per active month
        for mo in sorted(g.game_date.dt.to_period("M").unique()):
            for _ in range(rng.poisson(8)): note(m, mo.start_time + D(rng.integers(mo.days_in_month)), str(rng.choice(LAB, p=[.64, .2, .14, .02])))
    inj = incidents[incidents.kind == "injury"]
    for j in rng.permutation(len(inj))[:-(-3 * len(inj) // 4)]:  # 75% get a pre-IL note
        i = inj.iloc[j]; note(i.mlbam_id, i.il_date - D(rng.integers(1, 8)), str(rng.choice(LAB[2:])), i.body_part)
    for i in incidents.itertuples():
        if i.is_hero: note(i.mlbam_id, "2025-08-13", "red_flag", i.body_part)
        if i.sub_kind == "dead_arm_note": note(i.mlbam_id, i.il_date, "fatigue", text=DEAD)
    n = pd.DataFrame(rows, columns="note_ts athlete_id note_text true_category".split())
    n["note_ts"] += pd.to_timedelta(rng.integers(480, 1140, len(n)), "min")
    n.insert(2, "author_role", rng.choice(["athletic_trainer", "pitching_coach", "team_physician"], len(n)))
    n = n.sort_values(["note_ts", "athlete_id"], ignore_index=True)
    n.insert(0, "note_id", [f"N{i + 1:05d}" for i in n.index]); n["note_ts"] = n.note_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    return n.drop(columns="true_category"), n[["note_id", "true_category"]]

def make_il_events(incidents, roster):
    R, rows = roster.set_index("mlbam_id"), []
    for k, i in enumerate(incidents[incidents.kind == "injury"].itertuples()):
        r, d = R.loc[i.mlbam_id], int(i.il_days); w = f"Cascadia Ospreys %s {r.throws}HP {r.display_name} %s the {d}-day injured list."
        x = f" {'Right' if r.throws == 'R' else 'Left'} {i.body_part} {['strain', 'inflammation', 'tightness'][k % 3]}."
        rows += [(f"EV{2 * k + 1:03d}", r.athlete_id, str(i.il_date.date()), "placed", w % ("placed", "on") + x),
                 (f"EV{2 * k + 2:03d}", r.athlete_id, str((i.il_date + D(d)).date()), "activated", w % ("activated", "from"))]
    return pd.DataFrame(rows, columns="event_id athlete_id event_date action description".split())

def jl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); df.to_json(path, orient="records", lines=True)

def load(path):
    with open(path) as fh: return json.load(fh)

def main(argv):
    cfg, rows, hands, names = common.Cfg.from_args(argv), [], {}, set()
    L = cfg.landing
    for f in sorted(glob.glob(f"{L}/mlb_games/game_*.json")):
        g = load(f); rows += sc.extract_club_pitches(g, cfg.team_id, cfg.seed)
        h, n = sc.hands_and_names(g); hands.update(h); names |= n
    for f in glob.glob(f"{L}/mlb_transactions/txn_*.json"):  # every real name in the feeds is off-limits
        names |= {n for t in load(f)["transactions"] if (n := sc._g(t, "person", "fullName"))}
    p, rng = pd.DataFrame(rows), np.random.default_rng(cfg.seed)
    inc = sc.select_incidents(sc.appearances(p), rng)
    ros = make_roster(p.mlbam_id, rng, hands, names); ids = ros[["mlbam_id", "athlete_id"]]
    trk = sc.apply_precursors(p, inc); apps = sc.appearances(trk)
    wel, (notes, labels), il = make_wellness(ros, apps, inc, rng), make_notes(ros, apps, inc, rng), make_il_events(inc, ros)
    trk = trk.merge(ids).drop(columns="mlbam_id").assign(game_date=lambda x: x.game_date.dt.strftime("%Y-%m-%d"))
    gt = inc.merge(ids).drop(columns="mlbam_id")
    for c in ("true_onset_date", "il_date"): gt[c] = gt[c].dt.date
    man = pd.DataFrame([dict(seed=cfg.seed, version=sc.VERSION, n_pitches=len(trk), n_athletes=len(ros),
                             created_at=pd.Timestamp.now("UTC"))])
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    for t, df in (("incidents", gt), ("note_labels", labels), ("generator_manifest", man)):  # truth first, then feeds
        spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.fq("armguard_gt", t))
    for d in glob.glob(f"{L}/club_*"): shutil.rmtree(d)
    for pk, g in trk.groupby("game_pk"): jl(g, f"{L}/club_tracking/trk_{pk}.json")
    for df, path in ((wel, "club_wellness/wel"), (notes, "club_trainer_notes/notes"), (il, "club_medical/il_events"),
                     (ros.drop(columns="mlbam_id"), "club_roster/roster")): jl(df, f"{L}/{path}.json")
    n = {"tracking": len(trk), "athletes": len(ros), "wellness": len(wel), "notes": len(notes), "il_events": len(il)}
    common.write_evidence(cfg, "t2_simulator", "t2 - club systems simulator, sealed ground truth",
        pd.Series(n, name="rows").to_markdown() + "\n\n" + gt.groupby(["split", "kind"]).size().rename("n").reset_index().to_markdown(index=False)
        + f"\n\nhero athlete_id **{gt[gt.is_hero].athlete_id.iloc[0]}** (IL 2025-08-22) · seed {cfg.seed} · version {sc.VERSION}")

if __name__ == "__main__":
    main(sys.argv[1:])
