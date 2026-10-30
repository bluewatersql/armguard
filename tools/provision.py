"""Idempotent ArmGuard provisioning. Each step prints STEP <n> <name> OK|SKIP|FAIL."""
import argparse, json, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def db(profile, *args):
    return subprocess.run(["databricks", *args, "--profile", profile], cwd=ROOT, capture_output=True, text=True)


def step(n, name, status, detail=""):
    detail = re.sub(r"https?://\S+|\S+@\S+|[\w.-]+\.databricks\.com\b", "<redacted>", detail.strip())[-300:]  # no hosts/emails
    print(f"STEP {n} {name} {status}" + (f"  {detail}" if detail else ""), flush=True)
    return status != "FAIL"


def lakebase(a):
    """Lakebase project armguard (the CLI waits on the create operation); record its DB resource id."""
    P, st = "projects/armguard", "SKIP"
    if db(a.profile, "postgres", "get-project", P).returncode:
        cu = {"autoscaling_limit_min_cu": 0.5, "autoscaling_limit_max_cu": 2}
        spec = {"spec": {"display_name": "ArmGuard", "pg_version": 17, "default_endpoint_settings": cu}}
        r = db(a.profile, "postgres", "create-project", "armguard", "--json", json.dumps(spec))
        if r.returncode:
            return step(0, "lakebase", "FAIL", r.stderr)
        st = "OK"
    r = db(a.profile, "postgres", "list-databases", P + "/branches/production", "-o", "json")
    try:
        a.db_id = next(d["name"].rsplit("/", 1)[-1] for d in json.loads(r.stdout)
                       if d.get("status", {}).get("postgres_database") == "databricks_postgres")
    except (ValueError, KeyError, StopIteration):
        return step(0, "lakebase", "FAIL", "databricks_postgres not found " + r.stderr)
    return step(0, "lakebase", st, f"project armguard, lakebase_db_id {a.db_id}")


def catalog(a):
    if db(a.profile, "catalogs", "get", a.catalog).returncode == 0:
        return step(1, "catalog", "SKIP", "exists")
    if not a.create_catalog:
        return step(1, "catalog", "FAIL", "missing; pass --create-catalog")
    r = db(a.profile, "catalogs", "create", a.catalog)
    return step(1, "catalog", "OK" if r.returncode == 0 else "FAIL", r.stderr)


def overrides(a):
    p = ROOT / ".databricks" / "bundle" / a.target / "variable-overrides.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    new = {**cur, "catalog": a.catalog, **({"lakebase_db_id": a.db_id} if getattr(a, "db_id", None) else {})}
    if new == cur:
        return step(2, "overrides", "SKIP", "unchanged")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(new, indent=2) + "\n")
    return step(2, "overrides", "OK")


def bundle(a):
    for cmd in (["validate", "--strict"], ["deploy"]):
        r = db(a.profile, "bundle", *cmd, "-t", a.target)
        if r.returncode:
            return step(3, "bundle", "FAIL", r.stdout + r.stderr)
    return step(3, "bundle", "OK")


def bring_up(a):
    """Steps 4-9: full daily run (--skip-daily: SKIP), start the app, re-run t4/t8/t9 so grants find the app SP,
    restart (MASK_CHECK runs after the grants), smoke, then print the app state and URL."""
    for n, name, args in ((4, "daily", ["armguard_daily"]), (5, "app_start", ["armguard_app"]),
                          (6, "app_grants", ["armguard_daily", "--only", "t4_govern,t8_lakebase,t9_genie"])):
        if n == 4 and a.skip_daily:
            step(4, "daily", "SKIP", "--skip-daily")
            continue
        r = db(a.profile, "bundle", "run", *args, "-t", a.target)
        if not step(n, name, "FAIL" if r.returncode else "OK", r.stdout + r.stderr if r.returncode else ""):
            return False
    for cmd in ("stop", "start"):
        if (r := db(a.profile, "apps", cmd, "armguard")).returncode:
            return step(7, "app_restart", "FAIL", r.stderr)
    step(7, "app_restart", "OK")
    r = db(a.profile, "bundle", "run", "armguard_ops_tools", "--params", "mode=smoke", "-t", a.target)
    if not step(8, "smoke", "FAIL" if r.returncode else "OK", r.stdout + r.stderr if r.returncode else ""):
        return False
    for _ in range(30):  # the app status lags compute start
        r = db(a.profile, "apps", "get", "armguard", "-o", "json")
        try:
            d = json.loads(r.stdout)
        except ValueError:
            return step(9, "app", "FAIL", "apps get returned no JSON: " + r.stdout + r.stderr)
        if (state := d.get("app_status", {}).get("state")) == "RUNNING":
            break
        time.sleep(10)
    ok = step(9, "app", "OK" if state == "RUNNING" else "FAIL", f"state {state}")
    print("APP_URL", d.get("url"))  # console only: the URL holds the workspace host, never commit it
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--target", default="fevm")
    ap.add_argument("--create-catalog", action="store_true")
    ap.add_argument("--skip-daily", action="store_true", help="already-built workspace: skip the full daily run (step 4)")
    ap.add_argument("--teardown", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.teardown:
        print("teardown not implemented until Task 13")
        return 1
    return 0 if all(f(a) for f in (lakebase, catalog, overrides, bundle, bring_up)) else 1


if __name__ == "__main__":
    sys.exit(main())
