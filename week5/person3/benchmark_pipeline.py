"""
TrustGuard - Week 5 | Person 3: Benchmark & Evaluation Pipeline
----------------------------------------------------------------
Runs full evaluation of TrustGuard against the Week 4 adversarial suite.
- Computes: Precision, Recall, F1, Accuracy, AUC-ROC per hallucination category
- Confusion matrix (binary: hallucinated vs clean)
- Per-severity detection rate (CRITICAL/HIGH/MEDIUM/LOW)
- Threshold sensitivity analysis
- Outputs: week5_benchmark_report.json + week5_benchmark_plots/
  (primary handoff artifact for paper Section IV — Results)

All metrics reported at two levels:
  1. Binary detection (hallucinated=1, clean=0)
  2. Per-category breakdown (7 adversarial classes)

Publication target: IEEE Access / Springer LNCS format.
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week5_person3.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.P3")

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_LLM_FILE   = "week5_llm_outputs.json"      # from Person 1
INPUT_VAL_FILE   = "week5_validation_results.json" # from Person 3 Week 4
OUTPUT_JSON      = "week5_benchmark_report.json"
PLOT_DIR         = Path("week5_benchmark_plots")

# Risk score threshold: above this → predicted hallucinated
DEFAULT_THRESHOLD = 0.5

HALLUCINATION_CATEGORIES = [
    "over_permissive",
    "intent_flip",
    "wrong_port",
    "wrong_protocol",
    "missing_constraint",
    "scope_expansion",
    "security_downgrade",
]

CLEAN_LABEL = "clean"
SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


# ── Data Loader ───────────────────────────────────────────────────────────────
def load_benchmark_data(llm_path: str, val_path: str) -> pd.DataFrame:
    """
    Merge LLM outputs (Person 1) with validation results (Person 3 Week 4)
    on record_id to build the unified benchmark dataframe.
    """
    log.info(f"Loading LLM outputs: {llm_path}")
    with open(llm_path, "r") as f:
        llm_data = json.load(f)
    llm_records = llm_data.get("records", llm_data) if isinstance(llm_data, dict) else llm_data

    log.info(f"Loading validation results: {val_path}")
    with open(val_path, "r") as f:
        val_data = json.load(f)
    val_records = val_data.get("records", val_data) if isinstance(val_data, dict) else val_data

    # Build lookup: record_id → validation result
    val_lookup = {r.get("record_id"): r for r in val_records}

    rows = []
    for rec in llm_records:
        rid   = rec.get("record_id", "?")
        label = rec.get("ground_truth_label", "unknown")
        val   = val_lookup.get(rid, {})

        risk_score = (
            val.get("risk_score") or
            (val.get("validation") or {}).get("risk_aggregator", {}).get("final_risk_score") or
            0.0
        )

        compliance = (val.get("validation") or {}).get("compliance", {})
        max_severity = compliance.get("max_severity", "INFO")

        rows.append({
            "record_id":       rid,
            "label":           label,
            "is_hallucinated": 0 if label == CLEAN_LABEL else 1,
            "risk_score":      float(risk_score),
            "schema_valid":    rec.get("schema_valid", False),
            "max_severity":    max_severity,
            "confidence":      float((rec.get("parsed_policy") or {}).get("confidence", 0.0)),
        })

    df = pd.DataFrame(rows)
    log.info(f"Benchmark dataframe: {len(df)} records | "
             f"hallucinated={df['is_hallucinated'].sum()} | clean={(df['is_hallucinated']==0).sum()}")
    return df


# ── Binary Classification Metrics ────────────────────────────────────────────
def compute_binary_metrics(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Compute binary hallucination detection metrics at given risk threshold."""
    y_true = df["is_hallucinated"].values
    y_score = df["risk_score"].values
    y_pred  = (y_score >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # AUC — needs both classes present
    try:
        auc_roc = roc_auc_score(y_true, y_score)
        auc_pr  = average_precision_score(y_true, y_score)
    except ValueError:
        auc_roc = auc_pr = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold":         threshold,
        "precision":         round(float(prec), 4),
        "recall":            round(float(rec),  4),
        "f1_score":          round(float(f1),   4),
        "accuracy":          round(float(acc),  4),
        "auc_roc":           round(float(auc_roc), 4) if auc_roc else None,
        "auc_pr":            round(float(auc_pr),  4) if auc_pr  else None,
        "true_positives":    int(tp),
        "true_negatives":    int(tn),
        "false_positives":   int(fp),
        "false_negatives":   int(fn),
        "specificity":       round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0, 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 4),
    }


# ── Per-Category Metrics ──────────────────────────────────────────────────────
def compute_per_category_metrics(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Per hallucination category: precision, recall, F1, detection rate.
    Binary: category records vs clean records.
    """
    log.info("Computing per-category metrics...")
    results = {}
    clean_df = df[df["label"] == CLEAN_LABEL]

    for cat in HALLUCINATION_CATEGORIES:
        cat_df   = df[df["label"] == cat]
        if len(cat_df) == 0:
            log.warning(f"No records for category: {cat}")
            results[cat] = {"count": 0, "note": "no samples"}
            continue

        subset  = pd.concat([cat_df, clean_df])
        y_true  = subset["is_hallucinated"].values
        y_pred  = (subset["risk_score"].values >= threshold).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # Detection rate = fraction of category records flagged as hallucinated
        detection_rate = float((cat_df["risk_score"] >= threshold).mean())

        results[cat] = {
            "count":          len(cat_df),
            "precision":      round(float(prec), 4),
            "recall":         round(float(rec),  4),
            "f1_score":       round(float(f1),   4),
            "detection_rate": round(detection_rate, 4),
            "mean_risk":      round(float(cat_df["risk_score"].mean()), 4),
            "std_risk":       round(float(cat_df["risk_score"].std()),  4),
        }

    return results


# ── Threshold Sensitivity ─────────────────────────────────────────────────────
def threshold_sensitivity(df: pd.DataFrame) -> dict:
    """Sweep thresholds 0.1–0.9 and record F1, precision, recall at each."""
    log.info("Running threshold sensitivity analysis...")
    thresholds = np.arange(0.1, 1.0, 0.05).tolist()
    sweep = []
    for t in thresholds:
        m = compute_binary_metrics(df, threshold=round(t, 2))
        sweep.append({
            "threshold": round(t, 2),
            "f1":        m["f1_score"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "accuracy":  m["accuracy"],
        })
    # Best F1
    best = max(sweep, key=lambda x: x["f1"])
    return {"sweep": sweep, "best_threshold": best}


# ── Per-Severity Detection ────────────────────────────────────────────────────
def compute_severity_detection(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Detection rate breakdown by compliance severity tier."""
    results = {}
    for sev in SEVERITY_ORDER:
        sev_df = df[df["max_severity"] == sev]
        if len(sev_df) == 0:
            results[sev] = {"count": 0, "detection_rate": None}
            continue
        detected = (sev_df["risk_score"] >= threshold).sum()
        results[sev] = {
            "count":          len(sev_df),
            "detected":       int(detected),
            "detection_rate": round(float(detected / len(sev_df)), 4),
            "mean_risk":      round(float(sev_df["risk_score"].mean()), 4),
        }
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(df: pd.DataFrame, binary_metrics: dict, cat_metrics: dict,
             thresh_data: dict):
    PLOT_DIR.mkdir(exist_ok=True)
    y_true  = df["is_hallucinated"].values
    y_score = df["risk_score"].values

    # ── 1. ROC Curve ──────────────────────────────────────────────────────────
    if binary_metrics["auc_roc"] is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#2196F3", linewidth=2,
                label=f"AUC-ROC = {binary_metrics['auc_roc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title("TrustGuard: ROC Curve\nHallucination Detection (Binary)", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = PLOT_DIR / "roc_curve.png"
        plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
        log.info(f"Saved: {p}")

    # ── 2. Precision-Recall Curve ────────────────────────────────────────────
    if binary_metrics["auc_pr"] is not None:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(rec_arr, prec_arr, color="#4CAF50", linewidth=2,
                label=f"AUC-PR = {binary_metrics['auc_pr']:.4f}")
        ax.set_xlabel("Recall",    fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("TrustGuard: Precision-Recall Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = PLOT_DIR / "precision_recall_curve.png"
        plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
        log.info(f"Saved: {p}")

    # ── 3. Confusion Matrix ───────────────────────────────────────────────────
    y_pred = (y_score >= DEFAULT_THRESHOLD).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clean", "Hallucinated"],
                yticklabels=["Clean", "Hallucinated"],
                linewidths=0.5, ax=ax, cbar=False,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"TrustGuard: Confusion Matrix\n(threshold={DEFAULT_THRESHOLD})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = PLOT_DIR / "confusion_matrix.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # ── 4. Per-Category F1 Bar Chart ─────────────────────────────────────────
    cats   = [c for c in HALLUCINATION_CATEGORIES if cat_metrics.get(c, {}).get("count", 0) > 0]
    f1s    = [cat_metrics[c]["f1_score"]  for c in cats]
    recs   = [cat_metrics[c]["recall"]    for c in cats]
    precs  = [cat_metrics[c]["precision"] for c in cats]

    x = np.arange(len(cats))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w, precs, w, label="Precision", color="#2196F3", edgecolor="black", linewidth=0.6)
    ax.bar(x,     recs,  w, label="Recall",    color="#4CAF50", edgecolor="black", linewidth=0.6)
    ax.bar(x + w, f1s,   w, label="F1 Score",  color="#FF5722", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("TrustGuard: Detection Metrics per Hallucination Category",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "per_category_metrics.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # ── 5. Threshold Sensitivity ──────────────────────────────────────────────
    sweep = thresh_data["sweep"]
    ts    = [s["threshold"] for s in sweep]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, [s["f1"]        for s in sweep], "r-o",  label="F1",        linewidth=2, markersize=5)
    ax.plot(ts, [s["precision"] for s in sweep], "b-s",  label="Precision", linewidth=2, markersize=5)
    ax.plot(ts, [s["recall"]    for s in sweep], "g-^",  label="Recall",    linewidth=2, markersize=5)
    ax.axvline(thresh_data["best_threshold"]["threshold"], color="gray",
               linestyle="--", linewidth=1.5, label=f"Best F1 @ {thresh_data['best_threshold']['threshold']}")
    ax.set_xlabel("Risk Score Threshold", fontsize=12)
    ax.set_ylabel("Score",               fontsize=12)
    ax.set_title("TrustGuard: Threshold Sensitivity Analysis",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "threshold_sensitivity.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")

    # ── 6. Risk Score Distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    clean_scores = df[df["is_hallucinated"] == 0]["risk_score"]
    hall_scores  = df[df["is_hallucinated"] == 1]["risk_score"]
    ax.hist(clean_scores, bins=20, alpha=0.65, color="#4CAF50", label="Clean",        edgecolor="black", linewidth=0.5)
    ax.hist(hall_scores,  bins=20, alpha=0.65, color="#F44336", label="Hallucinated", edgecolor="black", linewidth=0.5)
    ax.axvline(DEFAULT_THRESHOLD, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold={DEFAULT_THRESHOLD}")
    ax.set_xlabel("Risk Score", fontsize=12)
    ax.set_ylabel("Count",      fontsize=12)
    ax.set_title("TrustGuard: Risk Score Distribution\nClean vs Hallucinated Policies",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = PLOT_DIR / "risk_score_distribution.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {p}")


# ── LaTeX Table Generator ─────────────────────────────────────────────────────
def generate_latex_table(cat_metrics: dict, binary_metrics: dict) -> str:
    """
    Generate LaTeX table for paper Section IV (Results).
    Compatible with IEEE two-column format.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TrustGuard: Per-Category Hallucination Detection Performance}",
        r"\label{tab:per_category_metrics}",
        r"\begin{tabular}{lccccr}",
        r"\hline",
        r"\textbf{Category} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{Det.Rate} & \textbf{N} \\",
        r"\hline",
    ]
    for cat in HALLUCINATION_CATEGORIES:
        m = cat_metrics.get(cat, {})
        if m.get("count", 0) == 0:
            continue
        cat_display = cat.replace("_", r"\_")
        lines.append(
            f"{cat_display} & {m['precision']:.3f} & {m['recall']:.3f} & "
            f"{m['f1_score']:.3f} & {m['detection_rate']:.3f} & {m['count']} \\\\"
        )
    lines += [
        r"\hline",
        f"\\textbf{{Overall (Binary)}} & {binary_metrics['precision']:.3f} & "
        f"{binary_metrics['recall']:.3f} & {binary_metrics['f1_score']:.3f} & "
        f"— & — \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_benchmark(llm_path: str = INPUT_LLM_FILE, val_path: str = INPUT_VAL_FILE):
    log.info("=" * 60)
    log.info("TrustGuard Week 5 | Person 3 | Benchmark & Evaluation")
    log.info("=" * 60)

    df = load_benchmark_data(llm_path, val_path)

    binary_metrics  = compute_binary_metrics(df)
    cat_metrics     = compute_per_category_metrics(df)
    thresh_data     = threshold_sensitivity(df)
    severity_data   = compute_severity_detection(df)
    latex_table     = generate_latex_table(cat_metrics, binary_metrics)

    plot_all(df, binary_metrics, cat_metrics, thresh_data)

    # Save latex table separately
    with open("week5_results_table.tex", "w") as f:
        f.write(latex_table)
    log.info("Saved: week5_results_table.tex")

    report = {
        "benchmark_run": {
            "timestamp":         datetime.utcnow().isoformat() + "Z",
            "total_records":     len(df),
            "hallucinated":      int(df["is_hallucinated"].sum()),
            "clean":             int((df["is_hallucinated"] == 0).sum()),
            "default_threshold": DEFAULT_THRESHOLD,
        },
        "binary_classification": binary_metrics,
        "per_category":          cat_metrics,
        "threshold_sensitivity": thresh_data,
        "severity_detection":    severity_data,
        "best_threshold":        thresh_data["best_threshold"],
        "latex_table_file":      "week5_results_table.tex",
        "plots": [str(p) for p in PLOT_DIR.glob("*.png")]
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info("Benchmark complete.")
    log.info(f"  Binary F1       : {binary_metrics['f1_score']}")
    log.info(f"  Binary AUC-ROC  : {binary_metrics['auc_roc']}")
    log.info(f"  Best threshold  : {thresh_data['best_threshold']['threshold']} "
             f"(F1={thresh_data['best_threshold']['f1']})")
    log.info(f"  Report          : {OUTPUT_JSON}")
    log.info(f"  LaTeX table     : week5_results_table.tex")
    log.info("=" * 60)

    return report


if __name__ == "__main__":
    run_benchmark()