"""Evidence tooling. --pull: copy the UC evidence volume into gitignored .superpowers/, scrub and print it.
--scrub FILE..: print local scratch files scrubbed. --gate: per-file 9k cap (hard) + evaluator read order."""
import argparse, json, pathlib, re, shutil, subprocess, sys
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAP, FILES, CHARS = 9000, 30, 90000  # evaluator: <= 9k chars/file (hard); reads <= 30 files / 90k chars
PATTERNS = [  # hosts first so a workspace id inside a host or app URL goes with it
    (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "<email>"),
    (r"(?:https?://)?[\w.-]+\.databricksapps\.com[^\s)|'\"<>]*", "<app-url>"),
    (r"(?:https?://)?[\w.-]+\.(?:cloud\.)?(?:azure)?databricks\.(?:com|net)[^\s)|'\"<>]*", "<host>"),
    (r"dapi[0-9a-f]{32}(?:-\d)?", "<token>"),
    (r"([?&]o=)\d+", r"\1<workspace-id>"),
]
CORE = ["databricks.yml", "resources/jobs.yml", "resources/pipeline.yml", "resources/app.yml", "src/common.py",
        "src/1_lakeflow/pipeline_bronze_silver.py", "src/1_lakeflow/pipeline_gold.py", "src/2_governance/govern.py",
        "src/3_ml/train.py", "src/4_genai/enrich.py", "src/5_lakebase/lakebase.py", "src/6_genie/genie.py",
        "src/7_app/app.py"]
SUPPORT = ["src/1_lakeflow/ingest_mlb.py", "src/1_lakeflow/simulate_club.py", "src/1_lakeflow/sim_core.py",
           "src/3_ml/score.py", "src/3_ml/holdout.py", "src/3_ml/validate.py", "src/4_genai/genai_eval.py",
           "src/6_genie/space.py", "src/6_genie/genie_ops.py", "src/7_app/cases.py", "src/7_app/requirements.txt"]
# docs, then one core file set per layer in layer order, then supporting files in layer order, tests/tools last
READ_ORDER = ["README.md", "EVIDENCE.md", "DESIGN.md", *CORE, *SUPPORT, "tests/test_armguard.py",
              "tests/test_cases.py", "tests/test_eval.py", "tools/provision.py", "tools/publish_evidence.py", ".gitignore"]


def scrub(s, literals=None):
    for pat, rep in PATTERNS:
        s = re.sub(pat, rep, s)
    for lit in sorted(literals or {}, key=len, reverse=True):
        s = s.replace(lit, literals[lit])
    return s


def prefix(order, sizes, files=FILES, chars=CHARS):
    """Greedy read-order prefix inside the budget; returns (files read, first file cut or None)."""
    take, total = [], 0
    for f in order:
        if len(take) + 1 > files or total + sizes[f] > chars:
            return take, f
        take.append(f)
        total += sizes[f]
    return take, None


def cli(profile, *args):
    r = subprocess.run(["databricks", *args, "--profile", profile], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"databricks {args[0]} failed: {scrub(r.stderr)[-300:]}")
    return r.stdout


def literals(profile, target):
    """Identifiers of the live workspace (CLI), its catalog (overrides) and any extras in a gitignored file."""
    d = json.loads(cli(profile, "auth", "describe", "-o", "json"))["details"]
    conf = {k: v.get("value") for k, v in d.get("configuration", {}).items()}
    lit = {urlparse(d["host"]).hostname: "<host>", conf.get("workspace_id"): "<workspace-id>",
           conf.get("account_id"): "<account-id>"}
    ov = ROOT / ".databricks" / "bundle" / target / "variable-overrides.json"
    lit[json.loads(ov.read_text()).get("catalog")] = "<catalog>"
    extra = ROOT / ".superpowers" / "scrub_literals.json"  # e.g. a previous workspace: {"<value>": "<label>"}
    lit.update(json.loads(extra.read_text()) if extra.exists() else {})
    return {k: v for k, v in lit.items() if k}


def pull(a):
    lit = literals(a.profile, a.target)
    cat = next(k for k, v in lit.items() if v == "<catalog>")
    raw, out = ROOT / ".superpowers" / "evidence-raw", ROOT / ".superpowers" / "evidence-scrubbed"
    for p in (raw, out):
        shutil.rmtree(p, ignore_errors=True)
    out.mkdir(parents=True)
    cli(a.profile, "fs", "cp", "-r", f"dbfs:/Volumes/{cat}/armguard_ops/evidence", str(raw))
    for f in sorted(raw.rglob("*.md")):
        text = scrub(f.read_text(encoding="utf-8"), lit)
        (out / f.name).write_text(text, encoding="utf-8")
        print(f"\n===== {f.name} =====\n{text}")
    shutil.rmtree(raw)  # only the scrubbed copy is kept (still gitignored)
    return 0


def gate(_):
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    sizes = {f: len((ROOT / f).read_text(encoding="utf-8")) for f in tracked}
    order = [f for f in READ_ORDER if f in sizes] + sorted(set(sizes) - set(READ_ORDER))
    take, cut = prefix(order, sizes)
    print("| # | file | chars | cumulative | read order |\n|---|---|---|---|---|")
    total = 0
    for i, f in enumerate(order, 1):
        total += sizes[f]
        tag = "in budget" if f in take else "CUT here" if f == cut else "past budget"
        print(f"| {i} | {f} | {sizes[f]:,} | {total:,} | {tag}{'' if f in READ_ORDER else ' (not in READ_ORDER)'} |")
    over = [f for f in order if sizes[f] > CAP]
    print(f"\nper-file cap {CAP:,}: {'FAIL ' + ', '.join(over) if over else 'PASS'} (max {max(sizes.values()):,})")
    print(f"whole repo: {len(order)} files, {total:,} chars -> " + ("OK" if len(order) <= FILES and total <= CHARS
          else f"OVER (reported, not hidden; budget {FILES} files / {CHARS:,} chars)"))
    print(f"read-order prefix: {len(take)} files, {sum(sizes[f] for f in take):,} chars; cut at {cut or '-'}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if "README.md" in sizes else ""
    if miss := [f for f in take if f not in readme]:
        print("WARN README 'Evaluator start here' does not list: " + ", ".join(miss))
    return 1 if over else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--scrub", nargs="+", metavar="FILE")
    ap.add_argument("--profile", help="required for --pull/--scrub; never auto-selected")
    ap.add_argument("--target", default="fevm")
    a = ap.parse_args()
    if (a.pull or a.scrub) and not a.profile:
        ap.error("--pull/--scrub need an explicit --profile")
    if a.scrub:
        lit = literals(a.profile, a.target)
        for f in a.scrub:
            print(f"\n===== {pathlib.Path(f).name} =====\n{scrub(pathlib.Path(f).read_text(encoding='utf-8'), lit)}")
        return 0
    return pull(a) if a.pull else gate(a) if a.gate else ap.error("pass --gate, --pull or --scrub")


if __name__ == "__main__":
    sys.exit(main())
