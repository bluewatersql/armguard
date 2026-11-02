# ArmGuard: pitcher arm-care early warning on Databricks

Data © MLB Advanced Media (statsapi.mlb.com) · players pseudonymized · medical data synthetic

## Problem and buyer
Pitcher injuries are the largest single source of lost value on an MLB roster, and the warning signs (pitch tracking, workload, wellness, trainer notes) sit in separate systems. The buyer is the President of Baseball Operations; the daily user is the Director of High Performance, who is judged on pitcher IL days, alert lead time and false alarms. ArmGuard gives the fictional **Cascadia Ospreys** one prioritized arm-care queue each morning, days before an IL placement, with a grounded brief and a tracked decision per flag. It is the "Databricks AI for Sports: Keep Players Healthy — injury mitigation" use case.

## Architecture
```
statsapi.mlb.com (real 2025)     club simulator (seed 4242; sealed truth -> armguard_gt)
         └────────► UC Volume landing ◄────────┘          Lakeflow Job armguard_daily
 t1-t3  SDP: Auto Loader -> bronze -> silver -> gold, expectations
 t4     UC: governed tags, ABAC mask on medical columns (probe), grants, lineage
 t5-t6  GBM (OOF thresholds) vs staff rule -> UC @champion; ai_classify note triage -> P1-P3
 t7     ai_extract on real IL text; grounded briefs, playbook-only actions
 t8     Lakebase: synced risk table + cases / case_events (OLTP, optimistic locks)
 t9     Genie "ArmGuard Analyst" + 12 unseen benchmarks
 t10    holdout backtest + null rate (gated); MLflow GenAI eval
 App "armguard" (Streamlit): Lakebase queue + actions, masked notes, Genie, Publish
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
| 17 | Lakeflow | `src/1_lakeflow/ingest_mlb.py` | §1 |

Past the budget, in layer order: `simulate_club`, `sim_core`, `score` (scoring + triage), `holdout`, `validate`, `genai_eval`, `space`, `genie_ops`, `cases`, tests, tools.

## Headline results (EVIDENCE.md)
- **Holdout:** the model catches 7/8 injuries (18-day median lead) vs the staff rule's 2/8, but fires on 2/4 hard negatives and 48% of healthy windows: **A4 MISS**, disclosed, not retuned.
- **Hero storyline:** pitcher *Daniel Stone* is P1 on 2025-08-15, 7 days before his IL date, because note triage flags a red_flag note.
- **GenAI and Genie:** body part 16/16; triage macro-F1 0.945; 51 briefs 100% grounded, no medical leak and playbook-only; Genie 11/12.
- **Governed, reprovisionable:** ABAC mask proven by a probe; Lakebase case loop rejects stale updates; one command rebuilds from zero.

## Run or reprovision
```
python tools/provision.py --profile <p> --catalog <c> [--create-catalog] [--skip-daily]
```
- **Prerequisites:** Databricks CLI ≥ 1.18, Python 3.10+, serverless jobs and SQL, Lakebase, FMAPI `databricks-claude-sonnet-5-5` and `-haiku-5-5`, a catalog you own (or `--create-catalog`); `--target` (default `fevm`) names a `databricks.yml` target.
- **Idempotent steps** (`STEP <n> <name> OK|SKIP|FAIL`): 0 Lakebase · 1 catalog · 2 gitignored overrides · 3 deploy · 4 daily run · 5 app start · 6 t4/t8/t9 app grants · 7 restart · 8 smoke · 9 app state.
- **Holdout validation is gated off by default.** `databricks bundle run armguard_daily --params validate=1` runs it once per config hash.
- **League IL KPIs freeze at first extraction.** To re-extract, run `DELETE FROM il_extractions`.
- **Tests:** `.venv/bin/python -m pytest tests -q`.

## Data and licensing
Games and transactions are a small slice of the public MLB Stats API for a non-commercial demo; raw payloads are never committed. Pitchers get seeded fake names, checked against every real player in the feeds. Precursors, wellness, trainer notes and club IL placements are synthetic: no real person's medical data appears.
