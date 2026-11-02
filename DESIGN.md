# DESIGN: decisions, disclosures, build process

One line per item. `Rnn` = a ruling in the build ledger; "user" = a decision the human made.

## Key decisions
- **Lakeflow end to end:** a Lakeflow Job lands MLB JSON and simulator feeds in a UC Volume; an SDP pipeline (Auto Loader → bronze → silver → gold, expectations, `full_refresh`) builds the features.
- **Bundle without `mode: development`** (unprefixed UC names); Lakebase project, synced table and Genie space are created idempotently with the SDK (no Beta DAB resources).
- **One command reprovisions** (`tools/provision.py`): no hand-copied ids or UI clicks; the app reads its runtime config from Delta `armguard_ops.app_config`.
- **Identity:** real 2025 MLB games for one club, shown as the fictional *Cascadia Ospreys*; pitchers get seeded fake names, rejected if they match any real player; all medical data is simulated.
- **Baseline to beat:** the staff rule `velo_delta <= -1.5` mph (last 3 vs prior 10 appearances).

## Task 0 spike outcomes
- PASS: Q1 serverless reaches statsapi.mlb.com, Q3 unprefixed bundle names, Q4 `singleVariantColumn` in SDP.
- Q2 `__file__` is undefined in `spark_python_task` → path bootstrap from `sys.argv[0]` (R5).
- Q5/Q6 account-wide `sensitivity`/`domain` tags have fixed values → own governed keys `armguard_sensitivity`, `armguard_tier` (R6); mask = ABAC `CREATE POLICY` on the tag, verified by a probe, with a legacy `SET MASK` fallback (R8).
- Q7 `ai_query` `responseFormat` fails on Claude 5.5 → JSON-only prompt parsed in Python; invalid output → template brief (R7). No sampling parameters.
- Q8 Lakebase SDK create + `.wait()`; synced table into a standard catalog. Q9 Genie eval-run API; `CAN_RUN` via the permissions API. Q10 serverless env `"3"`, inline app env.

## Why Lakebase
Staff act on alerts in an app: row-level transactions, an append-only `case_events` audit trail and optimistic locking (stale updates rejected) need OLTP. Scores stay governed in Delta and reach Postgres through a synced table; a snapshot job copies case history back to Delta for Genie.

## Why the model decides and GenAI explains
A GBM plus deterministic P1–P3 rules make every call, so each alert is reproducible and testable. GenAI extracts (IL body parts), triages notes (a red_flag note changes queue priority) and writes briefs grounded in the row's numbers, with actions limited to the 5-item playbook. The hero reaches P1 through red_flag triage while the model alone says ELEVATED (R32): the AI changes the workflow, not the score.

## Ground truth and holdout protocol
- The simulator seals truth in `armguard_gt` (owner only). Train IL dates 05-15..06-30, holdout 07-20..09-20; thresholds use train out-of-fold metrics only.
- Small N: 8 holdout injuries and 4 hard negatives; null pairs overlap in time and are not independent.
- **Holdout tally: 4 evaluations, all on the frozen model and thresholds.**
  - #1 original workspace: recall 7/8, FA 2/4, null 52.2%, hero P1; A4 MISS.
  - #2 (user: "re-run once, metric only"): metric corrected to the spec's in-window rule; recall 7/8, FA 2/4, null 47.8%; MISS.
  - #3 reproduction after the platform reset the workspace: identical model metrics, but hero P1 MISS (triage variance: the hero's red_flag note came back as soreness).
  - #4 after triage labels v3, chosen under a pre-registered dev/test protocol: identical model metrics, hero P1 PASS; A4 MISS.
- **User decision: accept the A4 MISS and disclose it rather than retune.** No threshold or rule was changed after seeing holdout results.
- `holdout_evaluations` is a per-workspace counter (the current workspace shows 2), so this list is the global tally.
- The OOF threshold rule still uses run-overlap detection, frozen since training; only the evaluation metric follows the spec (R49).
- **dead_arm asymmetry:** on the dead-arm hard negative the model is judged on P1, the staff rule on its alert runs.
- **R22 holdout peek:** a Task 4 sanity check saw holdout feature values (mean velo_delta at IL−2: −0.54 vs a −0.8 target); the simulator was not retuned.
- **R19:** the hero's decline was strengthened during simulator development (velo −2.2, spin −150 from IL−21, within spec ranges); the hero still counts in the holdout, flattering recall by at most 1/8. Realised decline 1.34 mph (R21).
- **Threshold determinism:** `GroupKFold` broke ties on equal athlete row counts by an unstable sort, so thresholds moved (0.13 → 0.17). Fix: greedy folds with an `athlete_id` tie-break. 0.13 / 0.078 now reproduce across runs and workspaces, but valid splits give HIGH from 0.06 to 0.21. The hero margin is thin: 0.0807 vs ELEVATED 0.078 (R45).

## Triage labels v3 (user: "try v3 with dev/test split")
- v3 = v1's none/fatigue/soreness labels plus v2's red_flag description. DEV (notes before 07-01) macro-F1 0.970 vs 0.903; TEST 0.928 vs 0.917; TEST red_flag recall 0.70 vs 0.57.
- It was prompted by the hero note, which is a holdout incident.
- The red_flag wording was derived from v2, which had seen all 939 notes, TEST included.
- The TEST macro-F1 margin (+0.011) is within v1's run-to-run noise (about ±0.015).
- v3 adds about 4–7 none → red_flag false positives.
- The experiment omitted `enableConfidenceScores`, which production passes.
- v2 alone failed its macro-F1 gate (0.878 < 0.891), so it was reverted.

## Genie benchmarks (frozen once written, R37)
- R39: the league "pitcher" definition was corrected to include `Pitcher`/`P` positions (323 → 467); this fixes ground truth, not Genie output.
- R41: Q12 wording changed to "decided cases only", because an undecided case has no answer.
- R54: Q12 is NEEDS_REVIEW; Genie's SQL differs only by the reference's `ROUND(…, 1)` and an extra `priority IS NOT NULL` filter. Left frozen: 11/12.

## Governance
- R26: extra app-SP grants (USE `armguard_gov`, SELECT `staff_roles`) are kept but unverified; the revoke test was blocked.
- R43: the app reads notes as its service principal (no on-behalf-of), so notes in the app are always masked; the medical/non-medical contrast is shown by the t4 probe.
- R46: no `GRANT EXECUTE ON MODEL`; the app never loads the model.

## Other disclosures
- R18: the role_change hard negative duplicates pitch rows so the outing reaches ≥ 60 pitches and flips RP → SP.
- R30: brief-count target relaxed to ≥ 40; the live eval has 51.
- R34: league IL KPIs freeze at first extraction; re-extract with `DELETE FROM il_extractions`.
- Reprovision run: t8's first attempt hit a platform SIGABRT and the automatic retry succeeded; `--teardown` is not implemented.

## Size budget
The evaluator reads up to 30 files and 90k characters, at most 9k per file. Every file is ≤ 9,000 characters (hard gate), but the repo is 33 files and about 190k characters. Many code files ran 2–3× over their per-file targets because correctness and demo reliability came first (user, R29). `tools/publish_evidence.py --gate` prints a read order: these three docs, then the core files of each layer. That prefix fits the budget; the rest is reported as past budget.

## AI-assisted build
Claude Code coordinated subagents: an implementer per task, an independent review of every task, fix rounds until clean. The human approved the concept, spec and plan gates and made the key decisions marked "user" here (accept A4 MISS, the single metric re-run, reproduction #3, labels v3, working demo over size caps). Hygiene: gitleaks secret scan on every commit; identifiers scrubbed by `publish_evidence.py`.
