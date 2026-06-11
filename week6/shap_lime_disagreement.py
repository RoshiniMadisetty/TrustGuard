"""
TrustGuard - Week 6 | Module 1: SHAP vs LIME Disagreement Detector
-------------------------------------------------------------------
Compares top-K features identified by SHAP (global) vs LIME (local)
for each policy record. Computes Jaccard-based agreement score and
flags records where the two XAI methods contradict each other.

Disagreement indicates the model relies on features that are
unstable under perturbation - a publishable signal of hallucination
uncertainty that neither SHAP nor LIME alone can surface.

Input  : week5_xai_report.json (Person 2 Week 5 output)
Output : week6_xai_disagreement.json
"""

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week6_disagreement.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.W6.Disagreement")

# -- Config --------------------------------------------------------------------
INPUT_FILE        = "week5_xai_report.json"
OUTPUT_FILE       = "week6_xai_disagreement.json"
PLOT_DIR          = Path("week6_plots")
TOP_K             = 5          # compare top-5 features from each method
DISAGREEMENT_THRESHOLD = 0.4  # Jaccard below this -> DISAGREEMENT


# -- Core: Jaccard Agreement ---------------------------------------------------
def jaccard_agreement(set_a: set, set_b: set) -> float:
    """
    Jaccard similarity between two feature sets.
    J(A,B) = |A n B| / |A u B|
    Range: 0.0 (no overlap) -> 1.0 (identical)
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return round(intersection / union, 4) if union > 0 else 0.0


def classify_agreement(score: float) -> str:
    if score >= 0.6:
        return "STRONG_AGREEMENT"
    elif score >= DISAGREEMENT_THRESHOLD:
        return "PARTIAL_AGREEMENT"
    else:
        return "DISAGREEMENT"


# -- Extract top-K features from SHAP global importance -----------------------
def get_shap_top_k(xai_report: dict, k: int = TOP_K) -> list:
    """
    Pull top-K features from SHAP global importance dict.
    Sorted descending by mean |SHAP| value.
    """
    importance = xai_report.get("shap", {}).get("global_feature_importance", {})
    if not importance:
        log.warning("No SHAP global importance found in XAI report.")
        return []
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in sorted_features[:k]]


# -- Extract top-K features from LIME per-sample -------------------------------
def get_lime_top_k(lime_entry: dict, k: int = TOP_K) -> list:
    """
    Pull top-K features from a single LIME explanation entry.
    LIME weights can be negative (inhibiting) - rank by abs value.
    """
    weights = lime_entry.get("lime_weights", {})
    if not weights:
        return []
    sorted_features = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

    # LIME feature names often contain condition strings like "confidence <= 0.75"
    # Strip to base feature name for fair comparison with SHAP
    clean = []
    for feat_str, _ in sorted_features[:k]:
        # Extract base feature name (before operators)
        base = feat_str.split(" ")[0].split(">")[0].split("<")[0].split("=")[0].strip()
        clean.append(base)
    return clean


# -- Per-Sample Disagreement Analysis -----------------------------------------
def analyze_disagreement(xai_report: dict) -> list:
    """
    For each LIME sample in the XAI report, compare its top-K features
    against the SHAP global top-K. Return per-sample disagreement records.
    """
    shap_top = get_shap_top_k(xai_report, TOP_K)
    if not shap_top:
        log.error("Cannot compute disagreement: SHAP top features missing.")
        return []

    shap_set = set(shap_top)
    lime_entries = xai_report.get("lime", {})
    results = []

    for sample_label, lime_data in lime_entries.items():
        lime_top = get_lime_top_k(lime_data, TOP_K)
        lime_set = set(lime_top)

        score  = jaccard_agreement(shap_set, lime_set)
        status = classify_agreement(score)

        overlap   = list(shap_set & lime_set)
        shap_only = list(shap_set - lime_set)
        lime_only = list(lime_set - shap_set)

        results.append({
            "sample":          sample_label,
            "record_id":       lime_data.get("record_id", "?"),
            "risk_score":      lime_data.get("risk_score", None),
            "ground_truth":    lime_data.get("ground_truth", "unknown"),
            "shap_top_k":      shap_top,
            "lime_top_k":      lime_top,
            "agreement_score": score,
            "status":          status,
            "overlap_features":   overlap,
            "shap_only_features": shap_only,
            "lime_only_features": lime_only,
            "interpretation": _interpret(status, overlap, shap_only, lime_only)
        })

        log.info(f"  {sample_label}: Jaccard={score:.3f} -> {status}")

    return results


def _interpret(status: str, overlap: list, shap_only: list, lime_only: list) -> str:
    """Human-readable interpretation for paper Table / Appendix."""
    if status == "STRONG_AGREEMENT":
        return (f"Both methods consistently identify {overlap} as primary risk drivers. "
                f"High explanation reliability.")
    elif status == "PARTIAL_AGREEMENT":
        return (f"Methods agree on {overlap} but diverge: SHAP additionally flags "
                f"{shap_only}, LIME flags {lime_only}. Moderate reliability.")
    else:
        return (f"Significant divergence. SHAP attributes risk to {shap_only}, "
                f"LIME to {lime_only}. Policy requires manual review - "
                f"model may be exploiting spurious features.")


# -- Aggregate Statistics ------------------------------------------------------
def aggregate_stats(records: list) -> dict:
    scores   = [r["agreement_score"] for r in records]
    statuses = [r["status"] for r in records]
    n        = len(records)

    return {
        "n_samples":              n,
        "mean_agreement":         round(float(np.mean(scores)), 4)  if scores else None,
        "std_agreement":          round(float(np.std(scores)),  4)  if scores else None,
        "min_agreement":          round(float(np.min(scores)),  4)  if scores else None,
        "max_agreement":          round(float(np.max(scores)),  4)  if scores else None,
        "strong_agreement_count": statuses.count("STRONG_AGREEMENT"),
        "partial_agreement_count":statuses.count("PARTIAL_AGREEMENT"),
        "disagreement_count":     statuses.count("DISAGREEMENT"),
        "disagreement_rate":      round(statuses.count("DISAGREEMENT") / n, 4) if n else 0,
        "disagreement_threshold": DISAGREEMENT_THRESHOLD,
        "top_k":                  TOP_K,
    }


# -- Plots ---------------------------------------------------------------------
def plot_disagreement(records: list, stats: dict):
    PLOT_DIR.mkdir(exist_ok=True)

    # -- Bar: Agreement score per sample ---------------------------------------
    labels = [r["sample"] for r in records]
    scores = [r["agreement_score"] for r in records]
    colors = ["#4CAF50" if s >= 0.6 else "#FF9800" if s >= DISAGREEMENT_THRESHOLD
              else "#F44336" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, scores, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(DISAGREEMENT_THRESHOLD, color="red",    linestyle="--", linewidth=1.5,
               label=f"Disagreement threshold ({DISAGREEMENT_THRESHOLD})")
    ax.axhline(0.6,                    color="green",  linestyle="--", linewidth=1.5,
               label="Strong agreement threshold (0.6)")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{score:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Jaccard Agreement Score", fontsize=12)
    ax.set_title("TrustGuard: SHAP vs LIME Feature Agreement\nPer-Sample Analysis",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "shap_lime_agreement.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # -- Pie: Status distribution -----------------------------------------------
    counts = [
        stats["strong_agreement_count"],
        stats["partial_agreement_count"],
        stats["disagreement_count"]
    ]
    status_labels = ["Strong Agreement", "Partial Agreement", "Disagreement"]
    colors_pie    = ["#4CAF50", "#FF9800", "#F44336"]
    non_zero = [(c, l, cl) for c, l, cl in zip(counts, status_labels, colors_pie) if c > 0]
    if non_zero:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([c for c, _, _ in non_zero],
               labels=[l for _, l, _ in non_zero],
               colors=[cl for _, _, cl in non_zero],
               autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 11})
        ax.set_title("TrustGuard: XAI Agreement Status Distribution",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        p = PLOT_DIR / "xai_agreement_distribution.png"
        plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
        log.info(f"Saved: {p}")


# -- Main ----------------------------------------------------------------------
def run_disagreement_analysis(input_path: str = INPUT_FILE) -> dict:
    log.info("=" * 60)
    log.info("TrustGuard Week 6 | Module 1 | SHAP-LIME Disagreement")
    log.info("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        xai_report = json.load(f)

    records = analyze_disagreement(xai_report)
    stats   = aggregate_stats(records)
    plot_disagreement(records, stats)

    output = {
        "module":    "shap_lime_disagreement",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config":    {"top_k": TOP_K, "disagreement_threshold": DISAGREEMENT_THRESHOLD},
        "summary":   stats,
        "per_sample_analysis": records
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    log.info(f"Mean agreement     : {stats['mean_agreement']}")
    log.info(f"Disagreement rate  : {stats['disagreement_rate']}")
    log.info(f"Output             : {OUTPUT_FILE}")
    log.info("=" * 60)
    return output


if __name__ == "__main__":
    run_disagreement_analysis()