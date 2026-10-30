"""Pure, seeded club simulator: GUMBO pitches -> club feeds + sealed ground truth."""
import hashlib
import numpy as np, pandas as pd
from faker import Faker

VERSION = "2.0"
D, T = (lambda n: pd.Timedelta(days=int(n))), pd.Timestamp
SPLITS = {"train": (T("2025-05-15"), T("2025-06-30")), "holdout": (T("2025-07-20"), T("2025-09-20"))}
NEG = {"train": "cold_weather role_change".split(), "holdout": "cold_weather cold_weather role_change dead_arm_note".split()}
NK = {"cold_weather": (2, 1.5), "role_change": (3, 1.2), "dead_arm_note": (1, 0)}
V, DROPS = "release_speed spin_rate release_z extension".split(), "velo_drop spin_drop rel_z_drop ext_drop".split()
BODY, LAB = "elbow shoulder forearm back oblique".split(), "none fatigue soreness red_flag".split()
DEAD = "Says arm feels dead and heavy after bullpen; velocity and command fine; no pain."
TPL = {k: t.split("|") for k, t in zip(LAB, (
    "No complaints.|Loose {s} {p}.|Normal recovery.|On schedule.|Checked {s} {p}: fine.|Long toss fine.",
    "Tired after big week.|Heavy {s} {p}, no pain.|Low energy.|Travel fatigue.|Tired {s} {p}, no pain.|Run down.",
    "Mild {s} {p} soreness.|Tight {s} {p}.|Tender {s} {p}.|Sore {s} {p} on long toss.|Achy {s} {p}.|Stiff {s} {p}.",
    "Sharp {s} {p} pain; shut down.|Tingling from {s} {p}.|Felt a pop in {s} {p}.|Pain in {s} {p} every throw."
    "|Grip down, {s} {p} pain.|Imaging for {s} {p}."))}
uid = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]

def _g(d, *ks):
    for k in ks: d = (d or {}).get(k)
    return d

def _ramp(d, on, il):  # concave 0->1 on [onset, IL)
    return np.where((d >= on) & (d < il), np.clip((d - on) / (il - on), 0, 1) ** .5, 0.0)

def extract_club_pitches(game, team_id, seed):
    gd, pk, out = game["gameData"], game["gamePk"], []
    t = str(_g(gd, "weather", "temp") or "").strip()
    temp, top = int(t) if t.lstrip("-").isdigit() else None, gd["teams"]["home"]["id"] == team_id
    for ab, pl in enumerate(game["liveData"]["plays"]["allPlays"]):
        for e in pl["playEvents"] if pl["about"]["isTopInning"] == top else []:
            q = e.get("pitchData") or {}
            if e.get("isPitch"): out.append(dict(pitch_uid=uid(f"{pk}-{ab}-{e['pitchNumber']}-{seed}"), game_pk=pk,
                game_date=gd["datetime"]["officialDate"], mlbam_id=pl["matchup"]["pitcher"]["id"],
                pitch_type=_g(e, "details", "type", "code"), release_speed=q.get("startSpeed"), spin_rate=_g(q, "breaks", "spinRate"),
                release_z=_g(q, "coordinates", "z0"), extension=q.get("extension"), temp_f=temp))
    return out

def hands_and_names(game):
    P = game["gameData"].get("players", {}).values()
    return {p["id"]: _g(p, "pitchHand", "code") or "R" for p in P}, {p["fullName"] for p in P if p.get("fullName")}

def appearances(pitches):
    d = pd.DataFrame(pitches).assign(game_date=lambda x: pd.to_datetime(x.game_date))
    d["fb"] = d.release_speed.where(d.pitch_type.isin(["FF", "SI", "FC"]))
    a = d.groupby(["mlbam_id", "game_date"]).agg(pitches=("fb", "size"), fb_velo=("fb", "mean"), v=("release_speed", "mean"))
    return a.assign(fb_velo=a.fb_velo.fillna(a.v), app_role=np.where(a.pitches >= 50, "SP", "RP")).drop(columns="v").reset_index()

def select_incidents(apps, rng):
    A = {int(m): sorted(g) for m, g in apps.groupby("mlbam_id").game_date}
    sp, ids, out, busy = (apps.app_role == "SP").groupby(apps.mlbam_id).mean(), sorted(A), [], {m: [] for m in A}
    elig = lambda m, t: sum(d < t - D(21) for d in A[m]) >= 5 and any(t - D(21) <= d < t for d in A[m])
    free = lambda m, a, b: all(b <= s - D(30) or a >= e + D(30) for s, e in busy[m])
    def injure(m, t, s, hero=False):
        on, days = t - D(rng.integers(10, 22)), 15 if hero else int(rng.choice([15, 15, 15, 60]))
        body = "elbow" if hero else str(rng.choice(BODY)); busy[m].append((on, t + D(days)))
        out.append((m, s, "injury", "", on, t, days, body, rng.uniform(1, 2.5), rng.uniform(80, 200), rng.uniform(.08, .2), .1, hero))
    H = T("2025-08-22")
    h = min(((min(abs(d - T("2025-08-12")) for d in A[m]), m) for m in ids if sp[m] > .5 and elig(m, H)), default=None)
    if not h: raise ValueError("insufficient eligible pitchers: hero")
    injure(h[1], H, "holdout", True)
    for s, (lo, hi) in SPLITS.items():
        n = int(s == "holdout")
        for m in map(int, rng.permutation(ids)):
            c = [t for t in (d + D(1) for d in A[m]) if lo <= t <= hi and elig(m, t) and free(m, t - D(21), t + D(60))]
            if c and n < 8: injure(m, c[rng.integers(len(c))], s); n += 1
        if n < 6: raise ValueError(f"insufficient eligible pitchers: {s} ({n} eligible)")
    for s, (lo, hi) in SPLITS.items():
        for k in NEG[s]:
            j = NK[k][0] - 1
            for m in map(int, rng.permutation(ids)):
                L = A[m]
                c = [i for i, d in enumerate(L) if lo <= d <= hi and 5 <= i < len(L) - j and free(m, d, L[i + j])
                     and (k != "role_change" or sp[m] < .5)]
                if c: break
            else: raise ValueError(f"insufficient eligible pitchers: {s} ({k})")
            i = c[rng.integers(len(c))]; busy[m].append((L[i], L[i + j]))
            out.append((m, s, "hard_negative", k, L[i], L[i], 0, "", NK[k][1], 0, 0, 0, False))
    df = pd.DataFrame(out, columns="mlbam_id split kind sub_kind true_onset_date il_date il_days body_part".split() + DROPS + ["is_hero"])
    df.insert(0, "incident_id", [f"inc_{i + 1:02d}" for i in df.index])
    return df.round(dict.fromkeys(DROPS, 3))

def apply_precursors(pitches, incidents):
    p = pitches.copy(); p["game_date"] = dd = pd.to_datetime(p.game_date)
    keep, dup = pd.Series(True, index=p.index), []
    for r in incidents.to_dict("records"):
        m, k, il = p.mlbam_id == r["mlbam_id"], r["sub_kind"], r["il_date"]
        if r["kind"] == "injury":
            for c, x in zip(V, DROPS): p[c] -= _ramp(dd, r["true_onset_date"], il) * m * r[x]
            keep &= ~(m & (dd >= il) & (dd < il + D(r["il_days"])))
            continue
        w = m & dd.isin(sorted(dd[m & (dd >= il)].unique())[:NK[k][0]])
        p.loc[w, "release_speed"] -= NK[k][1]
        if k == "cold_weather": p.loc[w, "temp_f"] = 44
        if k == "role_change":  # x1.6 pitches
            dup += [g.head(round(len(g) * .6)).assign(pitch_uid=g.pitch_uid.map(lambda u: uid(u + "r"))) for _, g in p[w].groupby(dd)]
    p = pd.concat([p[keep], *dup], ignore_index=True).astype({"temp_f": "Int64"})
    p["app_role"] = np.where(p.groupby(["mlbam_id", "game_date"]).pitch_uid.transform("size") >= 50, "SP", "RP")
    return p

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
        s = rng.poisson(1.5, n) + sum(np.round(_ramp(d, i.true_onset_date, i.il_date) * rng.uniform(2, 4))
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
        rows.append((T(day), aid[m], text or str(rng.choice(TPL[lab])).format(s=s, p=part or rng.choice(BODY)), lab))
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
