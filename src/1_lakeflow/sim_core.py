"""Pure, seeded club simulator: GUMBO pitches -> club feeds + sealed ground truth."""
import hashlib
import numpy as np, pandas as pd

VERSION = "2.0"
D, T = (lambda n: pd.Timedelta(days=int(n))), pd.Timestamp
SPLITS = {"train": (T("2025-05-15"), T("2025-06-30")), "holdout": (T("2025-07-20"), T("2025-09-20"))}
NEG = {"train": "cold_weather role_change".split(), "holdout": "cold_weather cold_weather role_change dead_arm_note".split()}
NK = {"cold_weather": (2, 1.5), "role_change": (3, 1.2), "dead_arm_note": (1, 0)}
V, DROPS = "release_speed spin_rate release_z extension".split(), "velo_drop spin_drop rel_z_drop ext_drop".split()
BODY = "elbow shoulder forearm back oblique".split()
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
    def injure(m, t, s, hero=False):  # hero is pinned: onset IL-21, 2.2 mph, 150 rpm
        on, days = t - D(21 if hero else rng.integers(10, 22)), 15 if hero else int(rng.choice([15, 15, 15, 60]))
        body = "elbow" if hero else str(rng.choice(BODY)); busy[m].append((on, t + D(days)))
        dr = [rng.uniform(1, 2.5), rng.uniform(80, 200), rng.uniform(.08, .2), .1]
        out.append((m, s, "injury", "", on, t, days, body, *([2.2, 150] if hero else dr[:2]), *dr[2:], hero))
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
    taken = set()  # hard negatives never share a game date (cold_weather now marks the whole game)
    for s, (lo, hi) in SPLITS.items():
        for k in NEG[s]:
            j = NK[k][0] - 1
            for m in map(int, rng.permutation(ids)):
                L = A[m]
                c = [i for i, d in enumerate(L) if lo <= d <= hi and 5 <= i < len(L) - j and free(m, d, L[i + j])
                     and (k != "role_change" or sp[m] < .5) and not taken & set(L[i:i + j + 1])]
                if c: break
            else: raise ValueError(f"insufficient eligible pitchers: {s} ({k})")
            i = c[rng.integers(len(c))]; busy[m].append((L[i], L[i + j])); taken |= set(L[i:i + j + 1])
            out.append((m, s, "hard_negative", k, L[i], L[i], 0, "", NK[k][1], 0, 0, 0, False))
    df = pd.DataFrame(out, columns="mlbam_id split kind sub_kind true_onset_date il_date il_days body_part".split() + DROPS + ["is_hero"])
    df.insert(0, "incident_id", [f"inc_{i + 1:02d}" for i in df.index])
    return df.round(dict.fromkeys(DROPS, 3))

def apply_precursors(pitches, incidents):
    p = pitches.copy(); p["game_date"] = dd = pd.to_datetime(p.game_date)
    keep, dup, cold = pd.Series(True, index=p.index), [], set()
    for r in incidents.to_dict("records"):
        m, k, il = p.mlbam_id == r["mlbam_id"], r["sub_kind"], r["il_date"]
        if r["kind"] == "injury":
            for c, x in zip(V, DROPS): p[c] -= _ramp(dd, r["true_onset_date"], il) * m * r[x]
            keep &= ~(m & (dd >= il) & (dd < il + D(r["il_days"])))
            continue
        w = m & dd.isin(sorted(dd[m & (dd >= il)].unique())[:NK[k][0]])
        p.loc[w, "release_speed"] -= NK[k][1]
        if k == "cold_weather": cold |= set(p.loc[w, "game_pk"])  # whole game is cold, not only his rows
        for _, g in p[w].groupby(dd) if k == "role_change" else []:  # RP -> SP: copies until >= 60 pitches, >= x1.6
            n = len(g); x = g.iloc[np.arange(max(60, -(-16 * n // 10)) - n) % n]
            dup.append(x.assign(pitch_uid=[uid(u + "r" * (1 + i // n)) for i, u in enumerate(x.pitch_uid)]))
    p = pd.concat([p[keep], *dup], ignore_index=True).astype({"temp_f": "Int64"})
    if cold: p.loc[p.game_pk.isin(cold), "temp_f"] = 44
    p["app_role"] = np.where(p.groupby(["mlbam_id", "game_date"]).pitch_uid.transform("size") >= 50, "SP", "RP")
    return p
