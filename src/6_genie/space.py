"""What Genie is taught (text instruction, example SQLs, joins, snippets) and build_space() -> serialized_space v2."""
import hashlib

RISK, DAILY, ILX, CASES = "{ops}.pitcher_risk_current", "{club}.gold_pitcher_daily_features", "{ops}.il_extractions", "{ops}.case_history"
TABLES = ("armguard.gold_appearances", "armguard.gold_pitcher_daily_features", "armguard.silver_roster", "armguard.silver_trainer_notes",
          "armguard_ops.case_history", "armguard_ops.il_extractions", "armguard_ops.pitcher_briefs", "armguard_ops.pitcher_risk_current")
EXAMPLES = [  # deliberately different questions from the benchmarks in genie.py
    ("What is the average fastball velocity change by pitcher role?",
     "SELECT app_role, ROUND(AVG(velo_delta), 2) AS avg_velo_delta FROM {club}.gold_appearances WHERE velo_delta IS NOT NULL GROUP BY app_role"),
    ("What is the latest date in the risk data?", f"SELECT MAX(as_of_date) AS latest_date FROM {RISK}"),
    ("Show me the brief for Paul Davis.",
     "SELECT r.display_name, b.headline, b.why, b.recommended_action FROM {ops}.pitcher_briefs b JOIN {club}.silver_roster r "
     f"ON b.athlete_id = r.athlete_id WHERE r.display_name = 'Paul Davis' AND b.as_of_date = (SELECT MAX(as_of_date) FROM {RISK})"),
    ("How many league IL placements were there by IL type?", f"SELECT il_type, COUNT(*) AS placements FROM {ILX} GROUP BY il_type ORDER BY placements DESC"),
    ("How many pitchers had soreness as their most serious trainer note last week?",
     f"SELECT COUNT(*) AS pitchers FROM {RISK} WHERE note_flag = 'soreness'"),
]
SAMPLES = ("Who needs attention today?", "Show me the brief for Charles Shea.", "Which relievers threw the most pitches in the last 28 days?")
VALUES = {"armguard_ops.case_history": ("action", "status"), "armguard_ops.il_extractions": ("body_part", "il_type", "position"),
          "armguard_ops.pitcher_risk_current": ("note_flag", "priority", "tier")}
JOINS = (("armguard.silver_roster", "armguard.gold_appearances"), ("armguard.silver_roster", "armguard.gold_pitcher_daily_features"),
         ("armguard_ops.pitcher_risk_current", "armguard_ops.pitcher_briefs"))


def nid(key):
    """Stable 32-hex id: re-runs keep ids, so eval results map back to question numbers."""
    return hashlib.md5(key.encode()).hexdigest()


def sub(sql, cat):
    return sql.replace("{club}", f"`{cat}`.armguard").replace("{ops}", f"`{cat}`.armguard_ops")


def text(t):
    return [s + "\n" for s in (
        "ArmGuard is the pitcher arm-care early-warning system of the Cascadia Ospreys (2025 MLB season). Pitcher names are pseudonymized; "
        "answer with display_name, not athlete_id.",
        "'Today', 'now', 'current' and 'latest date' mean MAX(as_of_date) in pitcher_risk_current (a single date); never use current_date().",
        "pitcher_risk_current: one row per eligible pitcher on the latest date with risk_score (model probability of an IL stint within 14 days), "
        "tier, note_flag, priority, recommended_action, top_drivers, baseline_flag, velo_delta, pitches_7d, acwr.",
        f"Tier: HIGH when risk_score >= {t['high']}, ELEVATED when risk_score >= {t['elevated']}, otherwise NORMAL "
        f"(thresholds from model_thresholds, model v{t['model_version']}).",
        "note_flag = the most serious trainer-note triage in the 7 days up to the as-of date, least to most serious: none, fatigue, soreness, "
        "red_flag. NULL = no note.",
        "Priority: P1 = red_flag note, or HIGH tier with soreness; P2 = HIGH tier, or ELEVATED tier with soreness or fatigue; P3 = ELEVATED tier; "
        "otherwise NULL. The arm-care queue = pitchers whose priority is not NULL; P1 is the most urgent.",
        "Recommended action = pitcher_risk_current.recommended_action, from the playbook: monitor, limit_pitches, extra_rest, bullpen_only, refer_medical.",
        "velo_delta = mean fastball mph of the last 3 appearances minus the mean of the 10 before (NULL with < 5 prior appearances); "
        "negative = velocity loss. Staff rule: alert when velo_delta <= -1.5 (baseline_flag).",
        "gold_appearances: one row per pitching appearance (game_date, pitches, app_role SP or RP, velo_delta); no names, join silver_roster on athlete_id.",
        "gold_pitcher_daily_features: one row per pitcher per calendar day (feature_date) with display_name; pitches_7d and pitches_28d = pitches "
        "in the 7 and 28 days ending on feature_date (inclusive); acwr = acute:chronic workload ratio = pitches_7d / (pitches_28d / 4); "
        "role_sp = 1 for starting pitchers, 0 for relievers. Use it for workload as of a given day.",
        "pitcher_briefs: AI-written briefs (headline, why, recommended_action) per pitcher and as_of_date; for current briefs use the latest date.",
        "il_extractions: real 2025 MLB injured-list placements league-wide (public transactions parsed by AI), one row per placement with player, "
        "position, il_type, body_part, side. A pitcher IL placement has position 'RHP' or 'LHP'. Club medical records are not in this space.",
        "case_history: arm-care cases from the case store, one row per case with priority, status (open, acknowledged, action_set, referred, "
        "cleared, placed_il), action, n_events, hours_to_decision (case creation to first decision; NULL while undecided).",
        "silver_trainer_notes.note_text is masked for users outside the medical staff; if it is masked, say so and do not guess.")]


def build_space(cat, t, bench, tables=TABLES):
    """bench: [(q_num, question, reference SQL)]. Ids are 32-hex; id-keyed lists sorted (cookbook section 4)."""
    fq, srt = (lambda x: f"{cat}.{x}"), (lambda xs, k="id": sorted(xs, key=lambda x: x[k]))
    cc = {k: [{"column_name": c, "enable_format_assistance": True, "enable_entity_matching": True} for c in v] for k, v in VALUES.items()}
    cc["armguard_ops.pitcher_briefs"] = [{"column_name": "facts_json", "exclude": True}]

    def join(left, right):
        a, b = left.split(".")[1], right.split(".")[1]
        return {"id": nid(f"join {a} {b}"), "left": {"identifier": fq(left), "alias": a}, "right": {"identifier": fq(right), "alias": b},
                "sql": [f"`{a}`.`athlete_id` = `{b}`.`athlete_id`", "--rt=FROM_RELATIONSHIP_TYPE_ONE_TO_MANY--"]}
    return {"version": 2,
            "config": {"sample_questions": srt([{"id": nid("sample " + q), "question": [q]} for q in SAMPLES])},
            "data_sources": {"tables": srt([{"identifier": fq(x), **({"column_configs": srt(cc[x], "column_name")} if x in cc else {})}
                                            for x in tables], "identifier")},
            "instructions": {
                "text_instructions": [{"id": nid("text"), "content": text(t)}],
                "example_question_sqls": srt([{"id": nid("example " + q), "question": [q], "sql": [sub(s, cat)]} for q, s in EXAMPLES]),
                "join_specs": srt([join(*j) for j in JOINS if {*j} <= {*tables}]),
                "sql_snippets": {
                    "measures": [{"id": nid("avg_velo_delta"), "alias": "avg_velo_delta", "display_name": "Average velocity change",
                                  "sql": ["AVG(gold_appearances.velo_delta)"], "synonyms": ["average velo delta", "mean velocity change"]}],
                    "filters": [{"id": nid("high_priority"), "display_name": "High priority",
                                 "sql": ["pitcher_risk_current.priority IN ('P1', 'P2')"], "synonyms": ["urgent", "high priority"]}]}},
            "benchmarks": {"questions": srt([{"id": nid(f"bench {i}"), "question": [q], "answer": [{"format": "SQL", "content": [sub(s, cat)]}]}
                                             for i, q, s in bench])}}
