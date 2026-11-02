# EVIDENCE: ArmGuard

All numbers are copied from job output in the live workspace; workspace ids and hosts are scrubbed. Sources: the UC volume `armguard_ops.evidence` (`tools/publish_evidence.py --pull`), job-run metadata, the SDP event log and read-only `COUNT(*)`s, in a workspace rebuilt from zero on 2026-11-01.

## Results at a glance (spec §13)
| # | Criterion | Real value | Result |
|---|---|---|---|
| A1 | Daily job, every task SUCCESS, row counts | run `128736621004166`: 12 tasks SUCCESS (t8 after a platform auto-retry), t10_validate gated off; counts §1 | PASS |
| A2 | SDP expectations from the event log | 5 expectations; `velo_plausible` dropped 3 of 22,070 pitches; all others 0 failed | PASS |
| A3 | Tags, mask probe, MASK_CHECK, lineage | ABAC policy (path b) probe `effective=True`; app SP sees `[REDACTED…`; 12 lineage rows | PASS |
| A4 | Holdout backtest (evaluation #4) | recall 7/8 PASS · median lead 18 d PASS · hard-negative false alarms 2/4 **MISS** · model beats rule **MISS** · null rate 47.8% vs 2.2% (reported) · hero P1 PASS | **MISS** |
| A5 | GenAI quality | body part 16/16 = 100%; triage macro-F1 0.945; 51 briefs: grounded, no-leak, playbook all 100% | PASS |
| A6 | Genie ≥ 10/12 unseen benchmarks | 11/12; Q12 NEEDS_REVIEW, reference SQL rounds (R54) | PASS |
| A7 | Lakebase loop | synced 13 = gold 13; smoke rejects a stale update; snapshot shows the 3 demo actions; Genie Q11 correct | PASS |
| A8 | App | RUNNING; `APP_ENV`, `MASK_CHECK`, 3× `CASE_UPDATE` log lines | PASS |
| A9 | Gate, secret scan, privacy | per-file ≤ 9,000 chars PASS; whole repo 33 files / ~190k chars OVER the 30 / 90k budget (reported); gitleaks clean; privacy grep clean | PARTIAL |
| A10 | Reprovision from zero | fresh fevm workspace after a platform reset: STEP 0–9 OK, no manual steps (user: counts as the proof) | PASS |

A4 is a recall vs false-alarm trade-off: 7/8 injuries caught (rule 2/8), but 2/4 hard negatives and ~half of healthy 3-week windows also alert. Model, thresholds and staff rule frozen (DESIGN.md).

## Run summary (scrubbed)
```
STEP 0 lakebase OK  project armguard, lakebase_db_id databricks-postgres
STEP 1 catalog SKIP  exists | STEP 2 overrides OK | STEP 3 bundle OK | STEP 4 daily OK
STEP 5 app_start OK | STEP 6 app_grants OK | STEP 7 app_restart OK | STEP 8 smoke OK
STEP 9 app OK  state RUNNING
```
| run id | what | result |
|---|---|---|
| 128736621004166 | daily run from zero (STEP 4) | t0–t7, t9, t10_gate, t10_genai_eval SUCCESS; t8 attempt 0 INTERNAL_ERROR (SIGABRT), auto-retry SUCCESS; t10_validate EXCLUDED |
| 910954703753794 | t4/t8/t9 for app SP grants (STEP 6) | SUCCESS |
| 49810023588883 | ops smoke (STEP 8) | attempt 0 INTERNAL_ERROR (SIGABRT), platform auto-retry SUCCESS |
| 405156101900698 | t6–t9 + t10_genai_eval, triage labels v3 | SUCCESS |
| 800103862519147 | holdout evaluation #4 (`validate=1`) | SUCCESS |
| 820778245496680 | snapshot (app Publish button) | SUCCESS |
| 565367944406204 | t9 Genie after demo actions | SUCCESS |

## 1. Lakeflow (t1–t3)
t1: 162 games fetched, 0 failed; 14,476 league transactions. t2 (seed 4242, v2.0): tracking 22,070 · athletes 33 · wellness 3,052 · notes 939 · IL events 32; truth: train 8 injuries + 2 hard negatives, holdout 8 + 4. SDP update `a259016e…` expectations:

| dataset | expectation | passed | failed |
|---|---|---|---|
| bronze_mlb_games / bronze_mlb_transactions | payload_present (fail) | 162 / 10 | 0 |
| silver_games | game_pk_present (drop) | 162 | 0 |
| silver_pitches | velo_plausible (drop) | 22,067 | 3 |
| silver_pitches | spin_plausible (warn) | 22,070 | 0 |

Rows: silver pitches 22,067 · wellness 3,052 · trainer_notes 939 · transactions (IL changes) 1,700; gold appearances 644 · pitcher_daily_features 6,138 · league_il_stints 846.

## 2. Unity Catalog (t4)
- 18 tier tags OK; `armguard_sensitivity=medical` on 5 columns (note_text, IL description, body_part, injury, side).
- `OK ABAC policy` → `PROBE path b effective=True`: without the medical role `'[REDACTED - medical staff only]'`; with it the text, e.g. `'Checked right oblique: fine.'` (synthetic).
- App SP grants OK (USE, SELECT on 10 tables, EXECUTE mask_medical); `case_history` SKIP until the first snapshot.
- Lineage (system tables): `silver_trainer_notes → note_triage`, `silver_transactions → il_extractions`, `pitcher_risk_current → pitcher_risk_synced`, …

## 3. ML (t5, t6, t10 validation)
t5: 924 train rows, 110 positives; OOF thresholds HIGH 0.13 / ELEVATED 0.078; OOF recall 1.00, fa/100 2.95 (staff rule 0.38 / 0.49); `injury_risk` v1 (MLflow run `c3f14433c8fb4362bc50e1a3ca1aec8b`), round-trip load OK.
t6 (as of 2025-08-15): 13 pitchers; HIGH P2 2 · ELEVATED P1 1, P2 1 · NORMAL 9. Hero A22 *Daniel Stone*: ELEVATED 0.081, red_flag note → **P1 PASS**.

Holdout evaluation #4 (run `800103862519147`, config `13575cbef60752ef`; detection = ≥ 2 consecutive alert days inside [IL−21, IL−1]):

| incident | A20 | A16 | A28 | A22 | A12 | A11 | A14 | A24 |
|---|---|---|---|---|---|---|---|---|
| IL date | 07-26 | 08-15 | 08-20 | 08-22 | 08-24 | 08-31 | 09-12 | 09-13 |
| model lead (d) | 18 | 10 | – | 16 | 18 | 11 | 21 | 21 |
| rule lead (d) | – | – | – | – | 9 | 17 | – | – |

| hard negative | model FA | rule FA |
|---|---|---|
| A31 cold_weather 08-05 · A26 cold_weather 09-03 | N · N | N · N |
| A17 dead_arm_note 08-07 (model judged on P1) | **Y** | N |
| A17 role_change 09-08 | **Y** | N |

| metric | model | staff rule |
|---|---|---|
| recall | 7/8 | 2/8 |
| median lead | 18.0 d | 13.0 d |
| hard-negative false alarms | 2/4 | 0/4 |
| null detection rate (46 pairs) | 47.8% | 2.2% |
| null, never-injured only (22) | 54.5% | 0.0% |

`A4 overall: MISS`. The local counter prints `holdout_evaluations: 2`; globally this is evaluation #4 (DESIGN.md).

## 4. GenAI (t7, t10 eval)
- `ai_extract` on real 2025 IL text: 846 league placements parsed, **467 pitcher IL placements**; by body part: other 113, elbow 109, shoulder 102, back 43, forearm 43, oblique 20.
- Briefs: 51 over 9 dates, 0 fallback. Hero: "Daniel Stone: ELEVATED tier, P1 with a red-flag note, so staff should review him now" → `refer_medical`.
- Eval (MLflow run `a81a80d1599a474b91f1a9d5d1ee2470`, `mlflow.genai.evaluate`): body part 16/16; triage (labels v3) macro-F1 0.945 over 939 notes, red_flag recall 25/32 (7 → soreness), 4 none → red_flag; brief_grounded, no_medical_leak and action_in_playbook all 100%.

## 5. Lakebase (t8, smoke, snapshot)
- t8: `Synced rows (Postgres) 13 vs gold rows (Delta) 13: PASS`; 0 cases opened this run (4 total); app SP grants OK.
- Smoke: `open -> acknowledged`, `acknowledged -> action_set`, `StaleUpdate caught`, `deleted 1`.
- Snapshot after the demo: 4 cases; `{'acknowledged': 1, 'cleared': 1, 'action_set': 1, 'open': 1}`; median 28.1 h to first decision.

## 6. Genie (t9)
Run `565367944406204`: `Score: 11/12 (target >= 10/12: PASS)`, eval-run API, benchmarks unseen by Genie. Q1–Q11 GOOD (Q8 = 467, matching t7; Q11 status counts match the snapshot). Q12 (average hours to first decision) NEEDS_REVIEW: Genie's SQL differs only by the reference's `ROUND(…, 1)` and an extra `priority IS NOT NULL`. Hero in Q2 and Q10: PASS.

## 7. App (log lines)
```
[APP] APP_ENV present=ARMGUARD_CATALOG,OPS_JOB_ID,LAKEBASE_ENDPOINT,DATABRICKS_WAREHOUSE_ID,PGUSER
[APP] MASK_CHECK value=[REDACTED - medical staf
[APP] CASE_UPDATE case_id=1 from=open to=action_set action=limit_pitches
[APP] CASE_UPDATE case_id=4 from=open to=acknowledged action=None
[APP] CASE_UPDATE case_id=2 from=open to=cleared action=None
```
