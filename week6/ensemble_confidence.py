"""
TrustGuard - Week 6 | Module 2: Ensemble Confidence Model
----------------------------------------------------------
Combines three independent confidence signals into a single
calibrated ensemble confidence score per policy record:

  1. model_confidence   - LLM's self-assessed confidence (0-1)
  2. validator_score    - normalized composite validation score (0-1)
  3. xai_agreement      - SHAP-LIME Jaccard agreement score (0-1)

Formula (weighted average, weights tunable):
  ensemble_confidence = w1 * model_confidence
                      + w2 * validator_score
                      + w3 * xai_agreement

Where w1=0.4, w2=0.3, w3=0.3 (sum=1.0)

Weights are justified in paper Section III-D:
  - LLM confidence weighted highest as primary signal
  - Validation and XAI equally weighted as corroborating signals

Outputs:
  - week6_ensemble_confidence.json
  - Confidence distribution plots
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week6_ensemble.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.W6.Ensemble")

# -- Config --------------------------------------------------------------------
INPUT_LLM_FILE   = "week5_llm_outputs.json"
INPUT_VAL_FILE   = "week5_validation_results.json"
INPUT_XAI_FILE   = "week6_xai_disagreement.json"
OUTPUT_FILE      = "week6_ensemble_confidence.json"
PLOT_DIR         = Path("week6_plots")

# Ensemble weights - must sum to 1.0
W_MODEL_CONFIDENCE = 0.40
W_VALIDATOR_SCORE  = 0.30
W_XAI_AGREEMENT    = 0.30

assert abs(W_MODEL_CONFIDENCE + W_VALIDATOR_SCORE + W_XAI_AGREEMENT - 1.0) < 1e-9, \
    "Ensemble weights must sum to 1.0"

# Severity -> penalty applied to validator_score
SEVERITY_PENALTY = {
    "CRITICAL": 0.40,
    "HIGH":     0.25,
    "MEDIUM":   0.10,
    "LOW":      0.05,
    "INFO":     0.00
}


# -- Signal Extractors ---------------------------------------------------------
def extract_model_confidence(llm_record: dict) -> float:
    """Extract LLM self-assessed confidence from parsed policy."""
    policy = llm_record.get("parsed_policy") or {}
    raw    = policy.get("confidence", None)
    if raw is None:
        # Fallback: schema_valid -> 0.5 base
        return 0.5 if llm_record.get("schema_valid", False) else 0.1
    try:
        return float(np.clip(float(raw), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.5


def extract_validator_score(val_record: dict) -> float:
    """
    Build a composite validator score from sub-validator results.
    Penalises by compliance severity and schema invalidity.

    Components:
      syntax_pass    -> 0.25
      semantic_sim   -> 0.25 (already 0-1)
      compliance     -> 0.25 (1 - severity_penalty)
      edge_case_pass -> 0.25 (1 - normalised edge case count)
    """
    val = val_record.get("validation") or {}

    # Syntax
    syntax_score = 0.25 if (val.get("syntax") or {}).get("valid", False) else 0.0

    # Semantic similarity
    raw_sem      = (val.get("semantic") or {}).get("similarity_score", 0.5)
    semantic_score = 0.25 * float(np.clip(raw_sem, 0.0, 1.0))

    # Compliance - penalise by max severity
    max_sev      = (val.get("compliance") or {}).get("max_severity", "INFO")
    penalty      = SEVERITY_PENALTY.get(max_sev, 0.0)
    compliance_score = 0.25 * (1.0 - penalty)

    # Edge cases - penalise by count (cap at 5)
    edge_count   = len((val.get("edge_case") or {}).get("triggered_cases", []))
    edge_score   = 0.25 * max(0.0, 1.0 - edge_count / 5.0)

    total = syntax_score + semantic_score + compliance_score + edge_score
    return round(float(np.clip(total, 0.0, 1.0)), 4)


def extract_xai_agreement(record_id: str, xai_report: dict,
                           fallback: float = 0.5) -> float:
    """
    Find the LIME entry closest to this record_id in the XAI disagreement report.
    Falls back to mean agreement score if record not found.
    """
    per_sample = xai_report.get("per_sample_analysis", [])

    # Direct match by record_id
    for entry in per_sample:
        if entry.get("record_id") == record_id:
            return float(entry.get("agreement_score", fallback))

    # Fallback: mean agreement across all samples
    mean = xai_report.get("summary", {}).get("mean_agreement")
    return float(mean) if mean is not None else fallback


# -- Ensemble Calculator -------------------------------------------------------
def compute_ensemble_confidence(model_conf: float,
                                 validator_score: float,
                                 xai_agreement: float) -> dict:
    """
    Compute weighted ensemble confidence and flag low-confidence records.
    """
    ensemble = (
        W_MODEL_CONFIDENCE * model_conf +
        W_VALIDATOR_SCORE  * validator_score +
        W_XAI_AGREEMENT    * xai_agreement
    )
    ensemble = round(float(np.clip(ensemble, 0.0, 1.0)), 4)

    # Confidence tier
    if ensemble >= 0.75:
        tier = "HIGH_CONFIDENCE"
    elif ensemble >= 0.50:
        tier = "MODERATE_CONFIDENCE"
    elif ensemble >= 0.30:
        tier = "LOW_CONFIDENCE"
    else:
        tier = "VERY_LOW_CONFIDENCE"

    return {
        "model_confidence":   round(model_conf,      4),
        "validator_score":    round(validator_score,  4),
        "xai_agreement":      round(xai_agreement,    4),
        "weighted_components": {
            "model_contribution":     round(W_MODEL_CONFIDENCE * model_conf,     4),
            "validator_contribution": round(W_VALIDATOR_SCORE  * validator_score, 4),
            "xai_contribution":       round(W_XAI_AGREEMENT    * xai_agreement,   4),
        },
        "ensemble_confidence": ensemble,
        "confidence_tier":    tier,
    }


# -- Batch Runner --------------------------------------------------------------
def run_ensemble_pipeline(llm_path:  str = INPUT_LLM_FILE,
                          val_path:  str = INPUT_VAL_FILE,
                          xai_path:  str = INPUT_XAI_FILE) -> dict:
    log.info("=" * 60)
    log.info("TrustGuard Week 6 | Module 2 | Ensemble Confidence")
    log.info("=" * 60)

    # Load inputs
    with open(llm_path, "r", encoding="utf-8") as f: llm_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f: val_data = json.load(f)
    with open(xai_path, "r", encoding="utf-8") as f: xai_data = json.load(f)

    llm_records = llm_data.get("records", llm_data) if isinstance(llm_data, dict) else llm_data
    val_records = val_data.get("records", val_data) if isinstance(val_data, dict) else val_data
    val_lookup  = {r.get("record_id"): r for r in val_records}

    results = []
    for rec in llm_records:
        rid = rec.get("record_id", "?")
        val = val_lookup.get(rid, {})

        model_conf    = extract_model_confidence(rec)
        val_score     = extract_validator_score(val)
        xai_agreement = extract_xai_agreement(rid, xai_data)

        conf_result = compute_ensemble_confidence(model_conf, val_score, xai_agreement)
        conf_result["record_id"]       = rid
        conf_result["ground_truth"]    = rec.get("ground_truth_label", "unknown")
        conf_result["schema_valid"]    = rec.get("schema_valid", False)

        results.append(conf_result)
        log.info(f"  {rid}: ensemble={conf_result['ensemble_confidence']} "
                 f"tier={conf_result['confidence_tier']}")

    # Aggregate stats
    ens_scores = [r["ensemble_confidence"] for r in results]
    tiers      = [r["confidence_tier"] for r in results]
    stats = {
        "n_records":              len(results),
        "mean_ensemble":          round(float(np.mean(ens_scores)), 4),
        "std_ensemble":           round(float(np.std(ens_scores)),  4),
        "min_ensemble":           round(float(np.min(ens_scores)),  4),
        "max_ensemble":           round(float(np.max(ens_scores)),  4),
        "high_confidence_count":      tiers.count("HIGH_CONFIDENCE"),
        "moderate_confidence_count":  tiers.count("MODERATE_CONFIDENCE"),
        "low_confidence_count":       tiers.count("LOW_CONFIDENCE"),
        "very_low_confidence_count":  tiers.count("VERY_LOW_CONFIDENCE"),
        "weights": {
            "model_confidence": W_MODEL_CONFIDENCE,
            "validator_score":  W_VALIDATOR_SCORE,
            "xai_agreement":    W_XAI_AGREEMENT
        }
    }

    _plot_confidence(results, stats)

    output = {
        "module":    "ensemble_confidence",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary":   stats,
        "records":   results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log.info(f"Mean ensemble confidence : {stats['mean_ensemble']}")
    log.info(f"High confidence records  : {stats['high_confidence_count']}")
    log.info(f"Output                   : {OUTPUT_FILE}")
    log.info("=" * 60)
    return output


# -- Plots ---------------------------------------------------------------------
def _plot_confidence(results: list, stats: dict):
    PLOT_DIR.mkdir(exist_ok=True)

    ens    = [r["ensemble_confidence"]    for r in results]
    model  = [r["model_confidence"]       for r in results]
    val    = [r["validator_score"]        for r in results]
    xai    = [r["xai_agreement"]          for r in results]
    labels = [r.get("ground_truth","?")   for r in results]

    # -- 1. Signal comparison scatter matrix -----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, x_vals, x_label in [
        (axes[0], model, "Model Confidence"),
        (axes[1], val,   "Validator Score"),
        (axes[2], xai,   "XAI Agreement"),
    ]:
        scatter_colors = ["#F44336" if l != "clean" else "#4CAF50" for l in labels]
        ax.scatter(x_vals, ens, c=scatter_colors, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="y=x")
        ax.set_xlabel(x_label,           fontsize=11)
        ax.set_ylabel("Ensemble Score",  fontsize=11)
        ax.set_title(f"{x_label} vs Ensemble", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    red_patch   = mpatches.Patch(color="#F44336", label="Hallucinated")
    green_patch = mpatches.Patch(color="#4CAF50", label="Clean")
    fig.legend(handles=[green_patch, red_patch], loc="upper center",
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("TrustGuard: Ensemble Confidence Signal Analysis",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    p = PLOT_DIR / "ensemble_signal_scatter.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- 2. Confidence tier distribution ---------------------------------------
    tier_order  = ["HIGH_CONFIDENCE","MODERATE_CONFIDENCE","LOW_CONFIDENCE","VERY_LOW_CONFIDENCE"]
    tier_colors = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336"]
    tier_counts = [stats.get(f"{t.lower()}_count", 0) for t in tier_order]
    tier_labels = [t.replace("_", "\n") for t in tier_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(tier_labels, tier_counts, color=tier_colors,
                  edgecolor="black", linewidth=0.7)
    for bar, count in zip(bars, tier_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Records", fontsize=12)
    ax.set_title("TrustGuard: Ensemble Confidence Tier Distribution",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "ensemble_confidence_tiers.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- 3. Stacked component breakdown (per record) ---------------------------
    rec_ids    = [r["record_id"] for r in results]
    comp_model = [r["weighted_components"]["model_contribution"]     for r in results]
    comp_val   = [r["weighted_components"]["validator_contribution"]  for r in results]
    comp_xai   = [r["weighted_components"]["xai_contribution"]       for r in results]
    x = np.arange(len(rec_ids))

    fig, ax = plt.subplots(figsize=(max(10, len(rec_ids)), 5))
    ax.bar(x, comp_model, label="Model (40%)",    color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.bar(x, comp_val,   bottom=comp_model,      label="Validator (30%)", color="#4CAF50", edgecolor="black", linewidth=0.5)
    bottom2 = [a + b for a, b in zip(comp_model, comp_val)]
    ax.bar(x, comp_xai,   bottom=bottom2,         label="XAI Agree (30%)", color="#FF9800", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(rec_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weighted Contribution", fontsize=11)
    ax.set_title("TrustGuard: Ensemble Confidence - Component Breakdown per Record",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "ensemble_component_breakdown.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")


if __name__ == "__main__":
    run_ensemble_pipeline()