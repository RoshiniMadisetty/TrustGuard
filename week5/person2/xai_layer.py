"""
TrustGuard - Week 5 | Person 2: XAI Explanation Layer
-------------------------------------------------------
Explainability module using SHAP + LIME on risk aggregator scores.
- Extracts numeric feature vectors from validated policies
- Trains a surrogate gradient boosting model on risk scores
- Applies SHAP TreeExplainer for global + local explanations
- Applies LIME for individual prediction explanations
- Outputs: week5_xai_report.json + week5_xai_plots/ (handoff to Person 3)

Publication context: IEEE/Springer — all results reproducible with fixed seeds.
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Suppress noisy warnings for clean publication output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week5_person2.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.P2")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
INPUT_FILE      = "week5_validation_results.json"   # output of Person 3 Week 4
OUTPUT_JSON     = "week5_xai_report.json"
PLOT_DIR        = Path("week5_xai_plots")
TOP_N_FEATURES  = 10   # for bar plots
LIME_SAMPLES    = 500  # perturbation samples


# ── Feature Extraction ────────────────────────────────────────────────────────
# Maps categorical policy fields to numeric features for model training

ACTION_MAP    = {"ALLOW": 0, "DENY": 1, "DROP": 2}
PROTOCOL_MAP  = {"TCP": 0, "UDP": 1, "ICMP": 2, "ANY": 3}
DIRECTION_MAP = {"INBOUND": 0, "OUTBOUND": 1, "BOTH": 2}
SEVERITY_MAP  = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

FEATURE_NAMES = [
    "action_enc",
    "protocol_enc",
    "direction_enc",
    "src_is_any",
    "dst_is_any",
    "src_port_is_any",
    "dst_port_is_any",
    "dst_port_numeric",
    "confidence",
    "priority_norm",
    "has_complete_cot",
    "reasoning_length",
    "syntax_valid",
    "semantic_score",
    "compliance_severity_enc",
    "edge_case_count",
]

def port_to_numeric(port) -> float:
    """Convert port value to float; 'ANY' → -1."""
    try:
        return float(port)
    except (TypeError, ValueError):
        return -1.0

def extract_features(record: dict) -> np.ndarray:
    """Extract numeric feature vector from a validation result record."""
    policy     = record.get("parsed_policy") or {}
    val        = record.get("validation") or {}
    syntax     = val.get("syntax", {})
    semantic   = val.get("semantic", {})
    compliance = val.get("compliance", {})
    edge       = val.get("edge_case", {})

    features = [
        ACTION_MAP.get(policy.get("action", ""), -1),
        PROTOCOL_MAP.get(policy.get("protocol", ""), -1),
        DIRECTION_MAP.get(policy.get("direction", ""), -1),
        1.0 if policy.get("src_ip") == "ANY" else 0.0,
        1.0 if policy.get("dst_ip") == "ANY" else 0.0,
        1.0 if policy.get("src_port") == "ANY" else 0.0,
        1.0 if policy.get("dst_port") == "ANY" else 0.0,
        port_to_numeric(policy.get("dst_port")),
        float(policy.get("confidence", 0.0)),
        float(policy.get("priority", 500)) / 1000.0,
        1.0 if policy.get("reasoning", "").count("Step") >= 3 else 0.0,
        min(float(len(policy.get("reasoning", ""))), 2000.0) / 2000.0,
        1.0 if syntax.get("valid", False) else 0.0,
        float(semantic.get("similarity_score", 0.0)),
        SEVERITY_MAP.get(compliance.get("max_severity", "INFO"), 0),
        float(len(edge.get("triggered_cases", []))),
    ]
    return np.array(features, dtype=np.float32)


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_data(path: str):
    """Load validation results and build feature matrix + risk score targets."""
    log.info(f"Loading data from {path}")
    with open(path, "r") as f:
        data = json.load(f)

    records = data.get("records", data) if isinstance(data, dict) else data

    X_rows, y_vals, meta = [], [], []

    for rec in records:
        risk = rec.get("risk_score")
        if risk is None:
            # Try nested path
            risk = (rec.get("validation") or {}).get("risk_aggregator", {}).get("final_risk_score")
        if risk is None:
            log.debug(f"Skipping record {rec.get('record_id')} — no risk score")
            continue

        try:
            feats = extract_features(rec)
            X_rows.append(feats)
            y_vals.append(float(risk))
            meta.append({
                "record_id":  rec.get("record_id", "?"),
                "label":      rec.get("ground_truth_label", "unknown"),
                "risk_score": float(risk)
            })
        except Exception as e:
            log.warning(f"Feature extraction failed for {rec.get('record_id')}: {e}")

    if not X_rows:
        raise ValueError("No valid records with risk scores found in input file.")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals,  dtype=np.float32)
    log.info(f"Feature matrix: {X.shape} | Targets: {y.shape}")
    return X, y, meta


# ── Surrogate Model ───────────────────────────────────────────────────────────
def train_surrogate(X: np.ndarray, y: np.ndarray):
    """
    Train gradient boosting surrogate model on risk scores.
    GBR chosen for: native SHAP TreeExplainer support, handles mixed features,
    no scaling required, publication-standard for XAI surrogates.
    """
    log.info("Training surrogate GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_SEED
    )
    model.fit(X, y)

    # 5-fold CV for publication reporting
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    log.info(f"Surrogate R² (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, cv_scores


# ── SHAP Global Explanations ──────────────────────────────────────────────────
def run_shap(model, X: np.ndarray, feature_names: list) -> dict:
    """Compute SHAP values — global importances + local waterfall for top/bottom samples."""
    log.info("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_samples, n_features)

    # Global: mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = dict(zip(feature_names, mean_abs_shap.tolist()))

    # Sort descending
    sorted_importance = dict(
        sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Plot 1: Global SHAP Bar ───────────────────────────────────────────────
    PLOT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = list(sorted_importance.keys())[:TOP_N_FEATURES]
    top_values   = [sorted_importance[f] for f in top_features]

    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top_features)))
    bars = ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP Value| (impact on risk score)", fontsize=12)
    ax.set_title("TrustGuard: Global Feature Importance (SHAP)\nTop Risk-Determining Policy Attributes",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path1 = PLOT_DIR / "shap_global_importance.png"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path1}")

    # ── Plot 2: SHAP Summary Beeswarm ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X,
                      feature_names=feature_names,
                      show=False, max_display=TOP_N_FEATURES,
                      plot_type="dot")
    plt.title("TrustGuard: SHAP Summary Plot (Feature Impact Distribution)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path2 = PLOT_DIR / "shap_summary_beeswarm.png"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path2}")

    # ── Local: waterfall for highest-risk sample ──────────────────────────────
    highest_risk_idx = int(np.argmax(np.abs(shap_values).sum(axis=1)))
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[highest_risk_idx],
            base_values=explainer.expected_value,
            data=X[highest_risk_idx],
            feature_names=feature_names
        ),
        show=False,
        max_display=10
    )
    plt.title("TrustGuard: SHAP Waterfall — Highest Risk Policy",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    path3 = PLOT_DIR / "shap_waterfall_high_risk.png"
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path3}")

    return {
        "global_feature_importance": sorted_importance,
        "shap_matrix_shape": list(shap_values.shape),
        "expected_value": float(np.asarray(explainer.expected_value).ravel()[0]),
        "plots": [str(path1), str(path2), str(path3)]
    }


# ── LIME Local Explanations ───────────────────────────────────────────────────
def run_lime(model, X: np.ndarray, feature_names: list, meta: list) -> dict:
    """Run LIME on 5 representative samples: top-2 high-risk, top-2 low-risk, 1 mid."""
    log.info("Running LIME local explanations...")

    risk_scores = np.array([m["risk_score"] for m in meta])
    sorted_idx  = np.argsort(risk_scores)

    # Select 5 representative indices
    n = len(sorted_idx)
    sample_indices = {
        "high_risk_1":  int(sorted_idx[-1]),
        "high_risk_2":  int(sorted_idx[-2]) if n > 1 else int(sorted_idx[-1]),
        "low_risk_1":   int(sorted_idx[0]),
        "low_risk_2":   int(sorted_idx[1]) if n > 1 else int(sorted_idx[0]),
        "mid_risk":     int(sorted_idx[n // 2]),
    }

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        mode="regression",
        random_state=RANDOM_SEED
    )

    lime_results = {}

    for label, idx in sample_indices.items():
        exp = explainer_lime.explain_instance(
            data_row=X[idx],
            predict_fn=model.predict,
            num_features=8,
            num_samples=LIME_SAMPLES
        )
        local_exp = exp.as_list()
        lime_results[label] = {
            "record_id":    meta[idx]["record_id"],
            "risk_score":   meta[idx]["risk_score"],
            "ground_truth": meta[idx]["label"],
            "lime_weights": {feat: float(w) for feat, w in local_exp},
            "prediction":   float(exp.predicted_value),
            "intercept":    float(exp.intercept[1] if hasattr(exp.intercept, '__len__') else exp.intercept)
        }

        # Plot LIME
        fig = exp.as_pyplot_figure()
        plt.title(f"TrustGuard LIME — {label} | Risk={meta[idx]['risk_score']:.3f}",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        path = PLOT_DIR / f"lime_{label}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        lime_results[label]["plot"] = str(path)
        log.info(f"  LIME {label}: {path}")

    return lime_results


# ── Hallucination Category XAI ────────────────────────────────────────────────
def xai_by_hallucination_category(X: np.ndarray, y: np.ndarray, meta: list,
                                   feature_names: list) -> dict:
    """
    Break down mean risk score and top SHAP feature per hallucination category.
    Publication table: Table IV in paper.
    """
    log.info("Computing per-hallucination-category XAI breakdown...")

    categories = {}
    for i, m in enumerate(meta):
        cat = m.get("label", "unknown")
        if cat not in categories:
            categories[cat] = {"indices": [], "risk_scores": []}
        categories[cat]["indices"].append(i)
        categories[cat]["risk_scores"].append(m["risk_score"])

    breakdown = {}
    for cat, info in categories.items():
        idxs      = info["indices"]
        risks     = info["risk_scores"]
        breakdown[cat] = {
            "count":          len(idxs),
            "mean_risk":      round(float(np.mean(risks)), 4),
            "std_risk":       round(float(np.std(risks)),  4),
            "max_risk":       round(float(np.max(risks)),  4),
            "min_risk":       round(float(np.min(risks)),  4),
        }

    # ── Bar chart: mean risk per category ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    cats  = list(breakdown.keys())
    means = [breakdown[c]["mean_risk"] for c in cats]
    stds  = [breakdown[c]["std_risk"]  for c in cats]

    palette = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    ax.bar(cats, means, yerr=stds, capsize=5, color=palette, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Hallucination Category", fontsize=12)
    ax.set_ylabel("Mean Risk Score", fontsize=12)
    ax.set_title("TrustGuard: Mean Risk Score by Hallucination Category\n(with std deviation)",
                 fontsize=13, fontweight="bold")
    ax.set_xticklabels(cats, rotation=30, ha="right")
    plt.tight_layout()
    path = PLOT_DIR / "risk_by_hallucination_category.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {path}")

    return breakdown


# ── Main ──────────────────────────────────────────────────────────────────────
def run_xai_pipeline(input_path: str = INPUT_FILE):
    log.info("=" * 60)
    log.info("TrustGuard Week 5 | Person 2 | XAI Explanation Layer")
    log.info("=" * 60)

    PLOT_DIR.mkdir(exist_ok=True)

    X, y, meta = load_data(input_path)
    model, cv_scores = train_surrogate(X, y)

    shap_results = run_shap(model, X, FEATURE_NAMES)
    lime_results = run_lime(model, X, FEATURE_NAMES, meta)
    cat_breakdown = xai_by_hallucination_category(X, y, meta, FEATURE_NAMES)

    report = {
        "xai_run": {
            "timestamp":           datetime.utcnow().isoformat() + "Z",
            "n_samples":           len(meta),
            "n_features":          len(FEATURE_NAMES),
            "feature_names":       FEATURE_NAMES,
            "surrogate_model":     "GradientBoostingRegressor",
            "surrogate_r2_cv_mean": round(float(cv_scores.mean()), 4),
            "surrogate_r2_cv_std":  round(float(cv_scores.std()),  4),
        },
        "shap": shap_results,
        "lime": lime_results,
        "hallucination_category_breakdown": cat_breakdown,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info(f"XAI pipeline complete.")
    log.info(f"  Surrogate R²  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    log.info(f"  SHAP plots    : {PLOT_DIR}/shap_*.png")
    log.info(f"  LIME plots    : {PLOT_DIR}/lime_*.png")
    log.info(f"  Category plot : {PLOT_DIR}/risk_by_hallucination_category.png")
    log.info(f"  Report        : {OUTPUT_JSON}")
    log.info("=" * 60)

    return report


if __name__ == "__main__":
    run_xai_pipeline()