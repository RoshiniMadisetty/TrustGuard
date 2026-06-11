"""
TrustGuard - Week 6 | Module 5: Safe / Review / Reject Decision Layer
----------------------------------------------------------------------
Final standalone decision module. Consumes ALL upstream signals and
produces a single, auditable verdict per policy record.

Decision inputs (in priority order):
  1. Adjusted risk score      (from edge case scoring)
  2. Ensemble confidence      (from ensemble confidence model)
  3. Calibrated thresholds    (from threshold calibration)
  4. XAI agreement status     (from SHAP-LIME disagreement module)
  5. Critical rule flags      (from edge case scoring)

Decision logic:
  +----------------------------------------------------------+
  | OVERRIDE rules (applied first, regardless of score):     |
  |   - Any CRITICAL edge case rule fired       -> REJECT     |
  |   - XAI status == DISAGREEMENT + risk > 0.4 -> REVIEW    |
  |                                                          |
  | THRESHOLD rules (applied after overrides):               |
  |   - adjusted_risk < safe_threshold          -> SAFE       |
  |   - adjusted_risk < review_threshold        -> REVIEW     |
  |   - adjusted_risk >= review_threshold       -> REJECT     |
  |                                                          |
  | CONFIDENCE modifier (softens/hardens verdict):           |
  |   - ensemble_confidence < 0.3 + SAFE        -> REVIEW    |
  |   - ensemble_confidence > 0.8 + REJECT      -> REJECT    |
  +----------------------------------------------------------+

Output per record:
  {
    "record_id":       "...",
    "decision":        "SAFE | REVIEW | REJECT",
    "decision_reason": "human-readable explanation",
    "adjusted_risk":   0.72,
    "ensemble_conf":   0.61,
    "xai_status":      "PARTIAL_AGREEMENT",
    "override_applied": false
  }

Outputs:
  - week6_decisions.json         (primary handoff for paper Table VI)
  - week6_decision_summary.json  (aggregate stats)
  - week6_plots/decision_*.png
"""

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week6_decision.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.W6.Decision")

# -- Inputs --------------------------------------------------------------------
INPUT_EDGE_FILE      = "week6_edge_case_scores.json"
INPUT_ENSEMBLE_FILE  = "week6_ensemble_confidence.json"
INPUT_THRESHOLD_FILE = "week6_calibrated_thresholds.json"
INPUT_XAI_FILE       = "week6_xai_disagreement.json"
INPUT_LLM_FILE       = "week5_llm_outputs.json"

OUTPUT_DECISIONS     = "week6_decisions.json"
OUTPUT_SUMMARY       = "week6_decision_summary.json"
PLOT_DIR             = Path("week6_plots")

DECISION_SAFE   = "SAFE"
DECISION_REVIEW = "REVIEW"
DECISION_REJECT = "REJECT"

DECISION_COLORS = {
    DECISION_SAFE:   "#4CAF50",
    DECISION_REVIEW: "#FF9800",
    DECISION_REJECT: "#F44336"
}


# -- Decision Engine -----------------------------------------------------------
def make_decision(adjusted_risk:    float,
                  ensemble_conf:    float,
                  xai_status:       str,
                  has_critical_rule: bool,
                  safe_threshold:   float,
                  review_threshold: float) -> tuple:
    """
    Apply decision logic in priority order.
    Returns (decision: str, reason: str, override_applied: bool)
    """

    # -- OVERRIDE 1: Critical edge case rule -----------------------------------
    if has_critical_rule:
        return (DECISION_REJECT,
                "Critical edge case rule triggered (e.g., ALLOW src=ANY dst=ANY or invalid port). "
                "Automatic REJECT regardless of risk score.",
                True)

    # -- OVERRIDE 2: XAI disagreement with elevated risk -----------------------
    if xai_status == "DISAGREEMENT" and adjusted_risk > 0.4:
        return (DECISION_REVIEW,
                f"SHAP-LIME disagreement detected (Jaccard < 0.4) with risk={adjusted_risk:.3f}. "
                "Explanation unreliable - escalated to REVIEW.",
                True)

    # -- THRESHOLD-BASED -------------------------------------------------------
    if adjusted_risk < safe_threshold:
        base_decision = DECISION_SAFE
        base_reason   = (f"Risk score {adjusted_risk:.3f} below SAFE threshold "
                         f"({safe_threshold:.3f}).")
    elif adjusted_risk < review_threshold:
        base_decision = DECISION_REVIEW
        base_reason   = (f"Risk score {adjusted_risk:.3f} in REVIEW band "
                         f"[{safe_threshold:.3f}, {review_threshold:.3f}).")
    else:
        base_decision = DECISION_REJECT
        base_reason   = (f"Risk score {adjusted_risk:.3f} exceeds REJECT threshold "
                         f"({review_threshold:.3f}).")

    # -- CONFIDENCE MODIFIER ---------------------------------------------------
    if ensemble_conf < 0.30 and base_decision == DECISION_SAFE:
        return (DECISION_REVIEW,
                base_reason + f" However, ensemble confidence is very low ({ensemble_conf:.3f} < 0.30) "
                "- upgraded to REVIEW for manual inspection.",
                False)

    return base_decision, base_reason, False


# -- Lookup Helpers ------------------------------------------------------------
def _build_edge_lookup(edge_data: dict) -> dict:
    records = edge_data.get("records", [])
    return {r["record_id"]: r for r in records}


def _build_ensemble_lookup(ens_data: dict) -> dict:
    records = ens_data.get("records", [])
    return {r["record_id"]: r for r in records}


def _get_xai_status_for_record(record_id: str, xai_data: dict) -> str:
    """Find XAI agreement status. Falls back to PARTIAL_AGREEMENT."""
    for entry in xai_data.get("per_sample_analysis", []):
        if entry.get("record_id") == record_id:
            return entry.get("status", "PARTIAL_AGREEMENT")
    return xai_data.get("summary", {}).get("status", "PARTIAL_AGREEMENT")


# -- Main Decision Runner ------------------------------------------------------
def run_decision_layer() -> dict:
    log.info("=" * 60)
    log.info("TrustGuard Week 6 | Module 5 | Decision Layer")
    log.info("=" * 60)

    # Load all upstream outputs
    with open(INPUT_EDGE_FILE,      "r") as f: edge_data  = json.load(f)
    with open(INPUT_ENSEMBLE_FILE,  "r") as f: ens_data   = json.load(f)
    with open(INPUT_THRESHOLD_FILE, "r", encoding="utf-8") as f: thresh_data = json.load(f)
    with open(INPUT_XAI_FILE,       "r") as f: xai_data   = json.load(f)
    with open(INPUT_LLM_FILE,       "r") as f: llm_data   = json.load(f)

    # Thresholds (primary = F1-optimal)
    primary = thresh_data.get("primary_thresholds", {})
    safe_t   = float(primary.get("safe_threshold",   0.25))
    review_t = float(primary.get("review_threshold", 0.50))

    log.info(f"Thresholds: SAFE < {safe_t} | REVIEW < {review_t} | REJECT >= {review_t}")

    edge_lookup = _build_edge_lookup(edge_data)
    ens_lookup  = _build_ensemble_lookup(ens_data)

    llm_records = llm_data.get("records", llm_data) if isinstance(llm_data, dict) else llm_data

    decisions = []

    for rec in llm_records:
        rid   = rec.get("record_id", "?")
        label = rec.get("ground_truth_label", "unknown")

        edge_rec = edge_lookup.get(rid, {})
        ens_rec  = ens_lookup.get(rid,  {})

        adjusted_risk    = float(edge_rec.get("adjusted_risk_score",  0.5))
        ensemble_conf    = float(ens_rec.get("ensemble_confidence",   0.5))
        has_critical     = bool(edge_rec.get("has_critical_rule",     False))
        xai_status       = _get_xai_status_for_record(rid, xai_data)

        decision, reason, override = make_decision(
            adjusted_risk, ensemble_conf, xai_status,
            has_critical, safe_t, review_t
        )

        record_out = {
            "record_id":        rid,
            "ground_truth":     label,
            "decision":         decision,
            "decision_reason":  reason,
            "override_applied": override,
            "signals": {
                "adjusted_risk_score":  round(adjusted_risk, 4),
                "ensemble_confidence":  round(ensemble_conf, 4),
                "xai_agreement_status": xai_status,
                "has_critical_rule":    has_critical,
                "confidence_tier":      ens_rec.get("confidence_tier", "UNKNOWN"),
                "triggered_rules":      [r["rule_id"] for r in edge_rec.get("triggered_rules", [])],
            }
        }
        decisions.append(record_out)

        icon = {"SAFE": "[OK]", "REVIEW": "[WARN]", "REJECT": "[FAIL]"}[decision]
        log.info(f"  {icon} {rid}: {decision} | risk={adjusted_risk:.3f} "
                 f"conf={ensemble_conf:.3f} | {reason[:60]}...")

    # -- Evaluation vs ground truth --------------------------------------------
    eval_stats = _evaluate_decisions(decisions)
    _plot_decisions(decisions, eval_stats)

    # -- Aggregate summary -----------------------------------------------------
    dec_counts = Counter(d["decision"] for d in decisions)
    overrides  = sum(1 for d in decisions if d["override_applied"])

    summary = {
        "module":           "decision_layer",
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "thresholds_used":  {"safe": safe_t, "review": review_t},
        "total_records":    len(decisions),
        "safe_count":       dec_counts[DECISION_SAFE],
        "review_count":     dec_counts[DECISION_REVIEW],
        "reject_count":     dec_counts[DECISION_REJECT],
        "override_count":   overrides,
        "evaluation":       eval_stats,
    }

    # Save decisions
    with open(OUTPUT_DECISIONS, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "decisions": decisions}, f, indent=2)

    # Save summary separately for easy reference
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info(f"SAFE   : {dec_counts[DECISION_SAFE]}")
    log.info(f"REVIEW : {dec_counts[DECISION_REVIEW]}")
    log.info(f"REJECT : {dec_counts[DECISION_REJECT]}")
    log.info(f"Overrides applied : {overrides}")
    log.info(f"Decisions output  : {OUTPUT_DECISIONS}")
    log.info("=" * 60)
    return {"summary": summary, "decisions": decisions}


# -- Evaluation ----------------------------------------------------------------
def _evaluate_decisions(decisions: list) -> dict:
    """
    Strict evaluation: only REJECT counts as a positive (hallucinated) prediction.

    Three-class mapping for publication:
      SAFE   predicted clean   -> correct for clean records (TN), wrong for hallucinated (FN)
      REVIEW predicted uncertain -> partial credit; counted as FN for recall purposes
      REJECT predicted hallucinated -> TP if actually hallucinated, FP if actually clean

    This is the conservative/strict metric. We also report a lenient metric
    where REVIEW+REJECT both count as flagged (lower precision, higher recall).
    Both are reported in the paper.
    """
    # Strict: only REJECT = positive prediction
    tp_s = fp_s = tn_s = fn_s = 0
    # Lenient: REVIEW or REJECT = positive prediction
    tp_l = fp_l = tn_l = fn_l = 0

    for d in decisions:
        gt      = d["ground_truth"]
        dec     = d["decision"]
        is_hall = gt not in ("correct", "clean")

        # Strict
        rejected = dec == DECISION_REJECT
        if is_hall and rejected:    tp_s += 1
        elif is_hall:               fn_s += 1
        elif not is_hall and rejected: fp_s += 1
        else:                       tn_s += 1

        # Lenient
        flagged = dec in [DECISION_REVIEW, DECISION_REJECT]
        if is_hall and flagged:     tp_l += 1
        elif is_hall:               fn_l += 1
        elif not is_hall and flagged: fp_l += 1
        else:                       tn_l += 1

    def metrics(tp, fp, tn, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        acc  = (tp + tn) / (tp+fp+tn+fn) if (tp+fp+tn+fn) > 0 else 0.0
        return {
            "true_positives":  tp, "false_positives": fp,
            "true_negatives":  tn, "false_negatives": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1_score":  round(f1,   4), "accuracy": round(acc, 4),
        }

    strict  = metrics(tp_s, fp_s, tn_s, fn_s)
    lenient = metrics(tp_l, fp_l, tn_l, fn_l)

    log.info(f"  Strict  (REJECT only) : P={strict['precision']}  "
             f"R={strict['recall']}  F1={strict['f1_score']}")
    log.info(f"  Lenient (REVIEW+REJECT): P={lenient['precision']} "
             f"R={lenient['recall']} F1={lenient['f1_score']}")

    # Primary metric for paper = strict
    result = strict.copy()
    result["strict"]  = strict
    result["lenient"] = lenient
    return result


# -- Plots ---------------------------------------------------------------------
def _plot_decisions(decisions: list, eval_stats: dict):
    PLOT_DIR.mkdir(exist_ok=True)

    # -- 1. Decision distribution pie -----------------------------------------
    counts = Counter(d["decision"] for d in decisions)
    labels = [k for k in [DECISION_SAFE, DECISION_REVIEW, DECISION_REJECT] if counts[k] > 0]
    values = [counts[k] for k in labels]
    colors = [DECISION_COLORS[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax.set_title("TrustGuard: Final Decision Distribution\nSAFE / REVIEW / REJECT",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = PLOT_DIR / "decision_distribution.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- 2. Decision vs ground truth heatmap -----------------------------------
    gt_labels  = sorted(set(d["ground_truth"] for d in decisions))
    dec_labels = [DECISION_SAFE, DECISION_REVIEW, DECISION_REJECT]
    matrix     = np.zeros((len(gt_labels), len(dec_labels)), dtype=int)

    for d in decisions:
        i = gt_labels.index(d["ground_truth"])
        j = dec_labels.index(d["decision"])
        matrix[i, j] += 1

    fig, ax = plt.subplots(figsize=(8, max(4, len(gt_labels) * 0.6 + 2)))
    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=dec_labels, yticklabels=gt_labels,
                linewidths=0.5, ax=ax, cbar=True,
                annot_kws={"size": 11})
    ax.set_xlabel("Predicted Decision", fontsize=12)
    ax.set_ylabel("Ground Truth Label", fontsize=12)
    ax.set_title("TrustGuard: Decision vs Ground Truth\nDetailed Breakdown",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = PLOT_DIR / "decision_vs_groundtruth.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- 3. Risk score by decision ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for dec in [DECISION_SAFE, DECISION_REVIEW, DECISION_REJECT]:
        risks = [d["signals"]["adjusted_risk_score"]
                 for d in decisions if d["decision"] == dec]
        if risks:
            ax.hist(risks, bins=15, alpha=0.65, color=DECISION_COLORS[dec],
                    label=dec, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Adjusted Risk Score", fontsize=12)
    ax.set_ylabel("Count",              fontsize=12)
    ax.set_title("TrustGuard: Risk Score Distribution by Decision",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "risk_by_decision.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- 4. Evaluation metrics bar ---------------------------------------------
    metrics = ["precision", "recall", "f1_score", "accuracy"]
    values  = [eval_stats[m] for m in metrics]
    colors  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(metrics, values, color=colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("TrustGuard: Decision Layer Evaluation Metrics",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "decision_evaluation_metrics.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")


if __name__ == "__main__":
    run_decision_layer()