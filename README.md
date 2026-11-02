# ArmGuard: pitcher arm-care early warning on Databricks

Data © MLB Advanced Media (statsapi.mlb.com) · players pseudonymized · medical data synthetic

## Problem and buyer
Pitcher injuries are the largest single source of lost value on an MLB roster, and the warning signs (pitch tracking, workload, wellness, trainer notes) sit in separate systems. The buyer is the President of Baseball Operations; the daily user, the Director of High Performance, is judged on IL days, alert lead time and false alarms. ArmGuard gives the fictional **Cascadia Ospreys** one prioritized arm-care queue each morning, days before an IL placement, with a grounded brief and a tracked decision per flag. It is the "Databricks AI for Sports: Keep Players Healthy — injury mitigation" use case.

## Architecture
```
statsapi.mlb.com (real 2025) + club simulator (sealed truth) -> UC Volume; Job armguard_daily
 t1-t3  SDP: Auto Loader -> bronze -> silver -> gold, expectations
 t4     UC: governed tags, ABAC mask on medical columns (probe), grants, lineage
 t5-t6  GBM (OOF thresholds) vs staff rule -> UC @champion; ai_classify note triage -> P1-P3
 t7     ai_extract on real IL text; grounded briefs, playbook-only actions
 t8     Lakebase: synced risk table + cases / case_events (OLTP, optimistic locks)
 t9     Genie "ArmGuard Analyst" + 12 unseen benchmarks
 t10    holdout backtest + null rate (gated); MLflow GenAI eval
 App (Streamlit): Lakebase queue + actions, masked notes, Genie, Publish
```

## Evaluator start here
Read in this order: the prefix that `tools/publish_evidence.py --gate` fits into the 30-file / 90k budget.

| # | Layer | Files | EVIDENCE |
|---|---|---|---|
| 1–3 | Docs | `README.md`, `EVIDENCE.md`, `DESIGN.md` | – |
| 4–7 | Bundle | `databricks.yml`, `resources/jobs.yml`, `resources/pipeline.yml`, `resources/app.yml` | Run summary |
| 8 | Shared | `src/common.py` (config, evidence writer, P1–P3 rules) | – |
| 9–10 | Lakeflow | `src/1_lakeflow/pipeline_bronze_silver.py`, `src/1_lakeflow/pipeline_gold.py` | §1 |
| 11 | Governance | `src/2_governance/govern.py` | §2 |
| 12 | ML | `src/3_ml/train.py` | §3 |
| 13 | GenAI | `src/4_genai/enrich.py` | §4 |
| 14 | Lakebase | `src/5_lakebase/lakebase.py` | §5 |
| 15 | Genie | `src/6_genie/genie.py` | §6 |
| 16 | App | `src/7_app/app.py` | §7 |

Past the budget, in order: `ingest_mlb`, `simulate_club`, `sim_core`, `score` (triage), `holdout`, `validate`, `genai_eval`, `space`, `genie_ops`, `cases`, `requirements.txt`, 3 tests, 2 tools, `.gitignore`.

## Headline results (EVIDENCE.md)
- **Holdout:** 7/8 injuries caught (18-day median lead) vs the staff rule's 2/8, but 2/4 hard negatives and 48% of healthy windows alert: **A4 MISS**, disclosed. **A9 PARTIAL:** per-file ≤ 9k passes; the repo is over the 30-file / 90k budget.
- **Hero:** *Daniel Stone* is P1 on 2025-08-15, 7 days before his IL date, via a red_flag triaged note.
- **GenAI and Genie:** body part 16/16; triage macro-F1 0.945; 51 briefs 100% grounded, no medical leak and playbook-only; Genie 11/12.
- **Governed, reprovisionable:** ABAC mask proven by a probe; stale case updates rejected; one command rebuilt it from zero in a fresh workspace (A10 PASS).

## Run or reprovision
```
python tools/provision.py --profile <p> --catalog <c> [--create-catalog] [--skip-daily]
```
- **Prerequisites:** Databricks CLI ≥ 1.18, Python 3.10+, serverless jobs and SQL, Lakebase, FMAPI `databricks-claude-sonnet-5-5` (`-haiku-5-5` only for `judge=1`), a catalog you own (or `--create-catalog`); `--target` (default `fevm`) names a `databricks.yml` target.
- **Idempotent steps** (`STEP <n> <name> OK|SKIP|FAIL`): 0 Lakebase · 1 catalog · 2 gitignored overrides · 3 deploy · 4 daily run · 5 app start · 6 t4/t8/t9 app grants · 7 restart · 8 smoke · 9 app state.
- **Holdout validation is gated off by default:** `databricks bundle run armguard_daily -t <target> --profile <p> --only t10_gate,t10_validate --params validate=1` (once per config hash; `force=1` re-runs it). Without `--only` it rebuilds everything first.
- **League IL KPIs freeze at first extraction.** To re-extract, run `DELETE FROM <catalog>.armguard_ops.il_extractions`.
- **Tests:** `python -m venv .venv && .venv/bin/pip install pytest pandas numpy faker==30.8.2 sqlalchemy tabulate`, then `.venv/bin/python -m pytest tests -q`.

## Data and licensing
A small slice of public MLB Stats API data, used for a non-commercial demo; raw payloads are never committed. Pitcher names are seeded fakes, checked against every real player. Precursors, wellness, notes and club IL placements are synthetic.
