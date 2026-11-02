"""Shared config, evidence writer and decision rules."""
import datetime as dt, os, re, sys
from dataclasses import dataclass, fields

PLAYBOOK = ("monitor", "limit_pitches", "extra_rest", "bullpen_only", "refer_medical")
WINDOWS = {3, 7, 10, 14, 21, 28, 30}
HERO = "A22"  # pinned storyline pitcher (t2)
# armguard_ops.case_history = HIST + display_name: snapshot() writes it; t8 sync() creates it empty so t4 grants and t9 attaches it
HIST = ("athlete_id STRING, case_id BIGINT, opened_for DATE, priority STRING, risk_score DOUBLE, status STRING, action STRING, pitch_limit INT, "
        "created_at TIMESTAMP, updated_at TIMESTAMP, n_events BIGINT, decided_at TIMESTAMP, hours_to_decision DOUBLE")

@dataclass(frozen=True)
class Cfg:
    catalog: str = "armguard_catalog_placeholder"
    team_id: int = 136
    season: int = 2025
    as_of_date: str = "2025-08-15"
    llm_endpoint: str = "databricks-claude-sonnet-5-5"
    judge_endpoint: str = "databricks-claude-haiku-5-5"
    seed: int = 4242
    presenter: str = ""
    mode: str = ""

    @classmethod
    def from_args(cls, argv):
        types = {f.name: f.type for f in fields(cls)}
        kw = {k: int(v) if types[k] in (int, "int") else v
              for k, v in (a[2:].split("=", 1) for a in argv if a[:2] == "--" and "=" in a) if k in types}
        return cls(**kw)

    landing = property(lambda s: f"/Volumes/{s.catalog}/armguard/landing")
    evidence_dir = property(lambda s: f"/Volumes/{s.catalog}/armguard_ops/evidence")

    def fq(self, schema, table):
        return f"`{self.catalog}`.`{schema}`.`{table}`"

def bootstrap():
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    sys.path[:0] = [here, os.path.dirname(here)]

def write_evidence(cfg, name, title, body, limit=3500):
    os.makedirs(cfg.evidence_dir, exist_ok=True)
    body = body if len(body) <= limit else body[:limit] + "\n…(truncated)"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    path = f"{cfg.evidence_dir}/{name}.md"
    with open(path, "w") as f:
        f.write(f"### {title}\n_job output, {stamp}_\n\n{body}\n")
    print(body)
    return path

def baseline_flag(velo_delta):
    return velo_delta is not None and velo_delta <= -1.5

def tier(score, high, elevated):
    return "HIGH" if score >= high else "ELEVATED" if score >= elevated else "NORMAL"

def priority(tier_, flag):
    if flag == "red_flag" or (tier_ == "HIGH" and flag == "soreness"):
        return "P1"
    if tier_ == "HIGH" or (tier_ == "ELEVATED" and flag in ("soreness", "fatigue")):
        return "P2"
    return "P3" if tier_ == "ELEVATED" else None

def finalize_brief(raw, facts):
    keys = ("headline", "why", "recommended_action", "confidence_note")
    if isinstance(raw, dict) and all(raw.get(k) for k in keys[:3]) and raw["recommended_action"] in PLAYBOOK:
        return {**{k: raw.get(k) or "" for k in keys}, "fallback": False}
    vd = facts.get("velo_delta")
    act = "refer_medical" if facts.get("note_flag") == "red_flag" else (
        "limit_pitches" if facts.get("tier") == "HIGH" else "monitor")
    why = (f"Fastball {vd:+.1f} mph vs baseline; " if vd is not None else "") + f"{facts.get('pitches_7d', 0)} pitches in 7 days."
    return {"headline": f"{facts['display_name']}: {facts.get('tier', 'REVIEW')} arm-care risk", "why": why,
            "recommended_action": act, "confidence_note": "Template brief (LLM unavailable).",
            "fallback": True}

_NUM = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?(?![A-Za-z\d])")

def numbers_grounded(text, facts):
    allowed = set(WINDOWS)
    for v in facts.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            allowed |= {abs(float(v)), round(abs(float(v)) * 100)}
        elif isinstance(v, str):
            allowed |= {float(n) for n in _NUM.findall(v)}
    return all(any(abs(float(n) - a) < 0.051 for a in allowed) for n in _NUM.findall(text or ""))
