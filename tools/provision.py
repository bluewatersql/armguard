"""Idempotent ArmGuard provisioning. Each step prints STEP <n> <name> OK|SKIP|FAIL."""
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def db(profile, *args):
    return subprocess.run(["databricks", *args, "--profile", profile], cwd=ROOT, capture_output=True, text=True)


def step(n, name, status, detail=""):
    print(f"STEP {n} {name} {status}" + (f"  {detail.strip()[:300]}" if detail else ""), flush=True)
    return status != "FAIL"


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
    new = {**cur, "catalog": a.catalog}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--target", default="fevm")
    ap.add_argument("--create-catalog", action="store_true")
    ap.add_argument("--teardown", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.teardown:
        print("teardown not implemented until Task 13")
        return 1
    return 0 if all(f(a) for f in (catalog, overrides, bundle)) else 1


if __name__ == "__main__":
    sys.exit(main())
