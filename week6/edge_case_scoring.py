"""
TrustGuard - Week 6 | Module 4: Edge Case Scoring Logic
--------------------------------------------------------
Applies targeted risk score adjustments for edge cases that the
base validation pipeline cannot catch through normal scoring.

Edge case rules (additive penalties / bonuses on base risk score):
  EC-01  Short response       : len(response) < 5 tokens      -> +20
  EC-02  Very low confidence  : model_confidence < 0.3        -> +15
  EC-03  Zero hallucinations  : hallucination_count == 0      -> -10
  EC-04  Missing reasoning    : CoT steps < 2                 -> +18
  EC-05  Port 0 or >65535     : invalid port value            -> +25
  EC-06  ANY+ANY+ALLOW        : src=ANY,dst=ANY,action=ALLOW  -> +40  (critical)
  EC-07  Negative priority    : priority <= 0                 -> +12
  EC-08  Empty policy fields  : any required field is ""      -> +20
  EC-09  Duplicate policy_id  : same id seen before           -> +10
  EC-10  Over-confident miss  : confidence > 0.9 + schema invalid -> +22

Final adjusted score is clamped to [0.0, 1.0].
All triggered rules are logged for audit trail.

Input  : week5_llm_outputs.json
Output : week6_edge_case_scores.json
"""

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week6_edgecase.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.W6.EdgeCase")

INPUT_LLM_FILE = "week5_llm_outputs.json"
INPUT_VAL_FILE = "week5_validation_results.json"
OUTPUT_FILE    = "week6_edge_case_scores.json"
PLOT_DIR       = Path("week6_plots")

# -- Edge Case Rule Definitions ------------------------------------------------
# Each rule: (id, description, penalty, severity)
EDGE_CASE_RULES = [
    ("EC-01", "Short/empty response (< 5 chars)",           +0.20, "HIGH"),
    ("EC-02", "Very low model confidence (< 0.30)",         +0.15, "HIGH"),
    ("EC-03", "Zero hallucination indicators (bonus)",      -0.10, "INFO"),
    ("EC-04", "Missing CoT reasoning (< 2 steps)",          +0.18, "HIGH"),
    ("EC-05", "Invalid port value (0 or > 65535)",          +0.25, "CRITICAL"),
    ("EC-06", "ALLOW with src=ANY and dst=ANY",             +0.40, "CRITICAL"),
    ("EC-07", "Negative or zero priority",                  +0.12, "MEDIUM"),
    ("EC-08", "Empty required field(s)",                    +0.20, "HIGH"),
    ("EC-09", "Duplicate policy_id",                        +0.10, "MEDIUM"),
    ("EC-10", "Over-confident + schema invalid",            +0.22, "HIGH"),
]
RULE_LOOKUP = {r[0]: r for r in EDGE_CASE_RULES}

REQUIRED_FIELDS = [
    "policy_id", "description", "action", "protocol",
    "src_ip", "dst_ip", "src_port", "dst_port",
    "direction", "priority", "reasoning", "confidence"
]


# -- Individual Rule Checkers --------------------------------------------------
def check_ec01(llm_rec: dict) -> bool:
    """Short or empty raw LLM response."""
    raw = llm_rec.get("raw_llm_output") or ""
    return len(str(raw).strip()) < 5


def check_ec02(policy: dict) -> bool:
    """Model confidence below 0.30."""
    try:
        return float(policy.get("confidence", 1.0)) < 0.30
    except (TypeError, ValueError):
        return False


def check_ec03(val_rec: dict) -> bool:
    """No hallucination indicators - give a small score bonus."""
    val          = val_rec.get("validation") or {}
    compliance   = val.get("compliance") or {}
    edge         = val.get("edge_case")  or {}
    syntax       = val.get("syntax")     or {}

    no_violations  = len(compliance.get("violations", [])) == 0
    no_edge_cases  = len(edge.get("triggered_cases", [])) == 0
    syntax_valid   = syntax.get("valid", False)
    return no_violations and no_edge_cases and syntax_valid


def check_ec04(policy: dict) -> bool:
    """CoT reasoning has fewer than 2 explicit steps."""
    reasoning = policy.get("reasoning", "")
    step_count = sum(1 for i in range(1, 6) if f"Step {i}" in reasoning)
    return step_count < 2


def check_ec05(policy: dict) -> bool:
    """Port value is 0, negative, or above 65535."""
    for port_key in ["src_port", "dst_port"]:
        val = policy.get(port_key)
        if val == "ANY":
            continue
        try:
            p = int(val)
            if p <= 0 or p > 65535:
                return True
        except (TypeError, ValueError):
            pass
    return False


def check_ec06(policy: dict) -> bool:
    """Critical: ALLOW with src_ip=ANY AND dst_ip=ANY."""
    return (
        policy.get("action")  == "ALLOW" and
        policy.get("src_ip")  == "ANY"   and
        policy.get("dst_ip")  == "ANY"
    )


def check_ec07(policy: dict) -> bool:
    """Priority is zero or negative."""
    try:
        return int(policy.get("priority", 1)) <= 0
    except (TypeError, ValueError):
        return False


def check_ec08(policy: dict) -> bool:
    """Any required field is present but empty string."""
    return any(
        policy.get(f) == "" or policy.get(f) is None
        for f in REQUIRED_FIELDS
    )


def check_ec09(policy_id: str, seen_ids: set) -> bool:
    """Policy ID has appeared before in this batch."""
    return policy_id in seen_ids


def check_ec10(policy: dict, schema_valid: bool) -> bool:
    """Over-confident (>0.9) but schema validation failed."""
    try:
        conf = float(policy.get("confidence", 0.0))
        return conf > 0.90 and not schema_valid
    except (TypeError, ValueError):
        return False


# -- Per-Record Edge Case Evaluation ------------------------------------------
def evaluate_edge_cases(llm_rec: dict, val_rec: dict,
                        base_risk: float, seen_ids: set) -> dict:
    """
    Run all 10 rules against a single record.
    Returns adjusted risk score + audit trail of triggered rules.
    """
    policy     = llm_rec.get("parsed_policy") or {}
    schema_ok  = llm_rec.get("schema_valid", False)
    policy_id  = policy.get("policy_id", "")

    triggered  = []
    total_adj  = 0.0

    checks = [
        ("EC-01", check_ec01(llm_rec)),
        ("EC-02", check_ec02(policy)),
        ("EC-03", check_ec03(val_rec)),
        ("EC-04", check_ec04(policy)),
        ("EC-05", check_ec05(policy)),
        ("EC-06", check_ec06(policy)),
        ("EC-07", check_ec07(policy)),
        ("EC-08", check_ec08(policy)),
        ("EC-09", check_ec09(policy_id, seen_ids)),
        ("EC-10", check_ec10(policy, schema_ok)),
    ]

    for rule_id, fired in checks:
        if fired:
            _, desc, penalty, severity = RULE_LOOKUP[rule_id]
            triggered.append({
                "rule_id":   rule_id,
                "description": desc,
                "adjustment": penalty,
                "severity":  severity
            })
            total_adj += penalty

    adjusted_risk = float(np.clip(base_risk + total_adj, 0.0, 1.0))

    # Track seen IDs for EC-09
    if policy_id:
        seen_ids.add(policy_id)

    return {
        "base_risk_score":     round(base_risk,     4),
        "total_adjustment":    round(total_adj,      4),
        "adjusted_risk_score": round(adjusted_risk,  4),
        "triggered_rules":     triggered,
        "rule_count":          len(triggered),
        "has_critical_rule":   any(r["severity"] == "CRITICAL" for r in triggered),
    }


# -- Batch Runner --------------------------------------------------------------
def run_edge_case_scoring(llm_path: str = INPUT_LLM_FILE,
                          val_path: str = INPUT_VAL_FILE) -> dict:
    log.info("=" * 60)
    log.info("TrustGuard Week 6 | Module 4 | Edge Case Scoring")
    log.info("=" * 60)

    with open(llm_path, "r", encoding="utf-8") as f: llm_data = json.load(f)
    with open(val_path,  "r") as f: val_data = json.load(f)

    llm_records = llm_data.get("records", llm_data) if isinstance(llm_data, dict) else llm_data
    val_records = val_data.get("records", val_data) if isinstance(val_data, dict) else val_data
    val_lookup  = {r.get("record_id"): r for r in val_records}

    seen_ids  = set()
    results   = []
    rule_freq = Counter()

    for rec in llm_records:
        rid       = rec.get("record_id", "?")
        val_rec   = val_lookup.get(rid, {})
        base_risk = float(
            (val_rec.get("validation") or {})
            .get("risk_aggregator", {})
            .get("final_risk_score", 0.5)
        )

        ec_result = evaluate_edge_cases(rec, val_rec, base_risk, seen_ids)

        for rule in ec_result["triggered_rules"]:
            rule_freq[rule["rule_id"]] += 1

        results.append({
            "record_id":       rid,
            "ground_truth":    rec.get("ground_truth_label", "unknown"),
            **ec_result
        })

        if ec_result["triggered_rules"]:
            log.info(f"  {rid}: base={base_risk:.3f} -> "
                     f"adj={ec_result['adjusted_risk_score']:.3f} "
                     f"rules={[r['rule_id'] for r in ec_result['triggered_rules']]}")

    # Summary
    adjustments = [r["total_adjustment"]    for r in results]
    base_scores = [r["base_risk_score"]     for r in results]
    adj_scores  = [r["adjusted_risk_score"] for r in results]

    summary = {
        "n_records":                len(results),
        "records_with_adjustments": sum(1 for r in results if r["rule_count"] > 0),
        "critical_rule_flags":      sum(1 for r in results if r["has_critical_rule"]),
        "mean_base_risk":           round(float(np.mean(base_scores)), 4),
        "mean_adjusted_risk":       round(float(np.mean(adj_scores)),  4),
        "mean_adjustment":          round(float(np.mean(adjustments)), 4),
        "rule_frequency":           dict(rule_freq.most_common()),
    }

    _plot_edge_cases(results, rule_freq)

    output = {
        "module":    "edge_case_scoring",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rules_applied": len(EDGE_CASE_RULES),
        "summary":   summary,
        "records":   results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log.info(f"Records adjusted    : {summary['records_with_adjustments']}/{summary['n_records']}")
    log.info(f"Critical flags      : {summary['critical_rule_flags']}")
    log.info(f"Mean risk shift     : {summary['mean_base_risk']:.4f} -> {summary['mean_adjusted_risk']:.4f}")
    log.info(f"Output              : {OUTPUT_FILE}")
    log.info("=" * 60)
    return output


# -- Plots ---------------------------------------------------------------------
def _plot_edge_cases(results: list, rule_freq: Counter):
    PLOT_DIR.mkdir(exist_ok=True)

    # -- Before vs After risk score scatter ------------------------------------
    base = [r["base_risk_score"]     for r in results]
    adj  = [r["adjusted_risk_score"] for r in results]
    has_crit = [r["has_critical_rule"] for r in results]
    colors = ["#F44336" if c else "#2196F3" for c in has_crit]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(base, adj, c=colors, alpha=0.7, edgecolors="black", linewidth=0.5, s=60)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="No change")
    ax.set_xlabel("Base Risk Score",     fontsize=12)
    ax.set_ylabel("Adjusted Risk Score", fontsize=12)
    ax.set_title("TrustGuard: Edge Case Score Adjustment\nBase vs Adjusted Risk",
                 fontsize=13, fontweight="bold")
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#F44336", label="Critical rule triggered"),
        mpatches.Patch(color="#2196F3", label="Normal adjustment"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="No change")
    ], fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "edge_case_score_adjustment.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- Rule frequency bar ----------------------------------------------------
    if rule_freq:
        rules  = list(rule_freq.keys())
        counts = list(rule_freq.values())
        sev_colors = {
            "CRITICAL": "#F44336", "HIGH": "#FF5722",
            "MEDIUM": "#FF9800",   "INFO": "#4CAF50"
        }
        bar_colors = [sev_colors.get(RULE_LOOKUP[r][3], "#9E9E9E") for r in rules]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(rules, counts, color=bar_colors, edgecolor="black", linewidth=0.7)
        for i, (rule, count) in enumerate(zip(rules, counts)):
            ax.text(i, count + 0.1, str(count), ha="center", fontsize=10)
        ax.set_ylabel("Times Triggered", fontsize=12)
        ax.set_title("TrustGuard: Edge Case Rule Trigger Frequency",
                     fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        legend_patches = [mpatches.Patch(color=c, label=s)
                          for s, c in sev_colors.items()]
        ax.legend(handles=legend_patches, fontsize=10, title="Severity")
        plt.tight_layout()
        p = PLOT_DIR / "edge_case_rule_frequency.png"
        plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
        log.info(f"Saved: {p}")


if __name__ == "__main__":
    run_edge_case_scoring()