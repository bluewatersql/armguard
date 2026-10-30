"""t1: land MLB Stats API data (c) MLB Advanced Media in the UC Volume; demo use."""
import calendar, datetime as dt, json, os, sys, time

BASE = "https://statsapi.mlb.com/api"

def final_game_pks(schedule):
    ok = lambda st: st.get("abstractGameState") == "Final" and st.get("detailedState") != "Postponed"
    return list(dict.fromkeys(g["gamePk"] for d in schedule.get("dates", []) for g in d.get("games", [])
                              if ok(g.get("status", {}))))

def month_windows(start, end):
    s, e, out = dt.date.fromisoformat(start), dt.date.fromisoformat(end), []
    while s <= e:
        last = min(s.replace(day=calendar.monthrange(s.year, s.month)[1]), e)
        out.append((s.isoformat(), last.isoformat()))
        s = last + dt.timedelta(1)
    return out

def needs_fetch(path):
    return not (os.path.exists(path) and os.path.getsize(path))

def get_json(ses, url, params=None):
    for i in range(4):
        r = ses.get(url, params=params, timeout=60)
        time.sleep(0.5 if r.ok else 2 ** i)
        if r.ok: return r.json()
    r.raise_for_status()

def land(ses, path, url, params=None, key=None):
    if not needs_fetch(path): return False
    d = get_json(ses, url, params)
    json.dump({key: d.get(key, [])} if key else d, open(path, "w"))
    return True

def main(argv):
    import requests
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
    from common import Cfg, write_evidence
    cfg, ses = Cfg.from_args(argv), requests.Session()
    gd, td = f"{cfg.landing}/mlb_games", f"{cfg.landing}/mlb_transactions"
    for d in (gd, td): os.makedirs(d, exist_ok=True)
    pks = final_game_pks(get_json(ses, f"{BASE}/v1/schedule",
                                  {"sportId": 1, "teamId": cfg.team_id, "season": cfg.season, "gameType": "R"}))
    got = skip = fail = n = 0
    for pk in pks:
        try:
            fetched = land(ses, f"{gd}/game_{pk}.json", f"{BASE}/v1.1/game/{pk}/feed/live")
            got, skip = got + fetched, skip + (not fetched)
        except Exception as e:
            fail += 1; print(f"WARN {pk}: {e!r:.80}")
    for a, b in month_windows(f"{cfg.season}-02-15", f"{cfg.season}-11-05"):
        p = f"{td}/txn_{a[:4]}{a[5:7]}.json"
        land(ses, p, f"{BASE}/v1/transactions", {"sportId": 1, "startDate": a, "endDate": b}, "transactions")
        n += len(json.load(open(p))["transactions"])
    write_evidence(cfg, "t1_ingest", "t1 - MLB Stats API to UC Volume",
        f"| games | fetched | skipped | failed | transactions |\n|---|---|---|---|---|\n"
        f"| {len(pks)} | {got} | {skip} | {fail} | {n} |")
    if fail > 5: raise RuntimeError(f"{fail} fetches failed")

if __name__ == "__main__":
    main(sys.argv[1:])
