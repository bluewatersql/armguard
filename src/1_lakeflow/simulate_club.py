"""t2: simulate club systems from landed GUMBO feeds; seal ground truth in armguard_gt."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
import common; common.bootstrap()
import glob, json, shutil
import numpy as np, pandas as pd
import sim_core as sc

def jl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True); df.to_json(path, orient="records", lines=True)

def main(argv):
    cfg, rows, hands, names = common.Cfg.from_args(argv), [], {}, set()
    L = cfg.landing
    for f in sorted(glob.glob(f"{L}/mlb_games/game_*.json")):
        g = json.load(open(f)); rows += sc.extract_club_pitches(g, cfg.team_id, cfg.seed)
        h, n = sc.hands_and_names(g); hands.update(h); names |= n
    p, rng = pd.DataFrame(rows), np.random.default_rng(cfg.seed)
    inc = sc.select_incidents(sc.appearances(p), rng)
    ros = sc.make_roster(p.mlbam_id, rng, hands, names); ids = ros[["mlbam_id", "athlete_id"]]
    trk = sc.apply_precursors(p, inc); apps = sc.appearances(trk)
    wel, (notes, labels), il = sc.make_wellness(ros, apps, inc, rng), sc.make_notes(ros, apps, inc, rng), sc.make_il_events(inc, ros)
    for d in glob.glob(f"{L}/club_*"): shutil.rmtree(d)
    trk = trk.merge(ids).drop(columns="mlbam_id").assign(game_date=lambda x: x.game_date.dt.strftime("%Y-%m-%d"))
    for pk, g in trk.groupby("game_pk"): jl(g, f"{L}/club_tracking/trk_{pk}.json")
    for df, path in ((wel, "club_wellness/wel"), (notes, "club_trainer_notes/notes"), (il, "club_medical/il_events"),
                     (ros.drop(columns="mlbam_id"), "club_roster/roster")): jl(df, f"{L}/{path}.json")
    gt = inc.merge(ids).drop(columns="mlbam_id")
    for c in ("true_onset_date", "il_date"): gt[c] = gt[c].dt.date
    man = pd.DataFrame([dict(seed=cfg.seed, version=sc.VERSION, n_pitches=len(trk), n_athletes=len(ros),
                             created_at=pd.Timestamp.now("UTC"))])
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    for t, df in (("incidents", gt), ("note_labels", labels), ("generator_manifest", man)):
        spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.fq("armguard_gt", t))
    n = {"tracking": len(trk), "athletes": len(ros), "wellness": len(wel), "notes": len(notes), "il_events": len(il)}
    common.write_evidence(cfg, "t2_simulator", "t2 - club systems simulator, sealed ground truth",
        pd.Series(n, name="rows").to_markdown() + "\n\n" + gt.groupby(["split", "kind"]).size().rename("n").reset_index().to_markdown(index=False)
        + f"\n\nhero athlete_id **{gt[gt.is_hero].athlete_id.iloc[0]}** (IL 2025-08-22) · seed {cfg.seed} · version {sc.VERSION}")

if __name__ == "__main__":
    main(sys.argv[1:])
