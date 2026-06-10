"""
TrustGuard — Person 2
Risk Engine v2 (Unified)

Combines:
  - SHAP explainability score
  - LIME explainability score
  - Person 3 validator score (syntax + semantic + compliance + edge case)
  - Model prediction confidence

Formula:
  Final Risk = 0.25 * SHAP_score
             + 0.25 * LIME_score
             + 0.35 * Validator_score   ← Person 3 contribution
             + 0.15 * Model_confidence_penalty

Classification:
  Final < 30   → SAFE
  30–60        → REVIEW
  60+          → REJECT

Run from anywhere:
  python risk_engine_v2.py
  OR
  cd person2_risk_engine && python risk_engine_v2.py
"""

import json
import os
import shap
import lime
import lime.lime_text
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime, UTC


# ──────────────────────────────────────────────
# PATHS — resolved relative to this script file
# so it works no matter where you run from
# ──────────────────────────────────────────────

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(THIS_DIR)   # firewall_validator/

P1_DATASET = os.path.join(ROOT_DIR, "person1_llm_pipeline", "data", "week4_final_dataset.json")
P3_SCORES  = os.path.join(ROOT_DIR, "person3_validator",    "outputs", "risk_aggregated_v2_results.json")
OUT_DIR    = THIS_DIR
OUT_CSV    = os.path.join(OUT_DIR, "risk_engine_v2_results.csv")
OUT_JSON   = os.path.join(OUT_DIR, "risk_engine_v2_results.json")

# ── Weights ───────────────────────────────────
WEIGHTS = {
    "shap":       0.25,
    "lime":       0.25,
    "validator":  0.35,   # Person 3 contribution
    "model_conf": 0.15,
}

# ── Classification thresholds (0–100 scale) ───
THRESHOLD_REVIEW = 30
THRESHOLD_REJECT = 60


# ──────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ──────────────────────────────────────────────

def load_data():
    if not os.path.exists(P1_DATASET):
        raise FileNotFoundError(
            f"Person 1 dataset not found at:\n  {P1_DATASET}\n"
            f"Make sure week4_final_dataset.json exists in person1_llm_pipeline/data/"
        )

    with open(P1_DATASET) as f:
        raw = json.load(f)

    # Build P3 scores lookup
    p3_lookup = {}
    if os.path.exists(P3_SCORES):
        with open(P3_SCORES) as f:
            p3_data = json.load(f)
        for rule in p3_data["rules"]:
            p3_lookup[rule["pair_id"]] = rule
        print(f"  ✓ Loaded {len(p3_lookup)} P3 validator scores from:")
        print(f"    {P3_SCORES}")
    else:
        print(f"  ⚠  P3 scores not found — validator score will be 0 for all rules")
        print(f"     Expected: {P3_SCORES}")
        print(f"     Run person3_validator/risk_aggregator_v2.py first")

    records = []
    for pair in raw["pairs"]:
        rule    = pair.get("generated_rule") or {}
        pid     = pair["pair_id"]
        p3      = p3_lookup.get(pid, {})
        p3_comp = p3.get("component_scores", {})

        records.append({
            "pair_id":              pid,
            "requirement":          pair["requirement"],
            "label":                pair["label"],
            "hallucination_type":   pair["hallucination_type"],
            "label_confidence":     pair["label_confidence"],
            "security_impact":      pair.get("security_impact", "None"),
            "compliance_violation": ", ".join(pair.get("compliance_violation", [])),
            "parse_success":        pair["generation_metadata"]["parse_success"],
            "action":               rule.get("action",           "unknown"),
            "protocol":             rule.get("protocol",         "unknown"),
            "source":               rule.get("source",           "unknown"),
            "destination":          rule.get("destination",      "unknown"),
            "destination_port":     str(rule.get("destination_port", "unknown")),
            # Person 3 scores (0–1 scale)
            "p3_syntax_score":      p3_comp.get("syntax",     0.0),
            "p3_semantic_score":    p3_comp.get("semantic",   0.0),
            "p3_compliance_score":  p3_comp.get("compliance", 0.0),
            "p3_edge_score":        p3_comp.get("edge_case",  0.0),
            "p3_weighted_risk":     p3.get("weighted_risk",   0.0),
            "p3_classification":    p3.get("classification",  "unknown"),
            "p3_halluc_types":      ", ".join(p3.get("hallucination_types_detected", [])),
        })

    df = pd.DataFrame(records)
    print(f"  ✓ Loaded {len(df)} rules from Person 1 dataset")
    print(f"  Label distribution:")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"    {lbl:<15} {cnt}")
    return df, p3_lookup


# ──────────────────────────────────────────────
# STEP 2 — TRAIN MODEL
# ──────────────────────────────────────────────

def train_model(df):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0))
    ])
    pipeline.fit(df["requirement"], df["label"])
    print(f"  ✓ Model trained. Classes: {list(pipeline.classes_)}")
    return pipeline


# ──────────────────────────────────────────────
# STEP 3 — SHAP SCORING
# ──────────────────────────────────────────────

def run_shap(df, pipeline):
    print("  Running SHAP (may take ~30s)...")
    explainer   = shap.Explainer(pipeline.predict_proba,
                                  shap.maskers.Text(r"\W+"))
    shap_values = explainer(df["requirement"].tolist())

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        sv = shap_values[i]

        if sv.values.ndim > 1:
            magnitudes = np.abs(sv.values).sum(axis=-1)
        else:
            magnitudes = np.abs(sv.values)

        raw_conf = float(max(magnitudes)) if len(magnitudes) else 0.0
        pct      = round(min(raw_conf / 0.4, 1.0) * 100, 2)

        if len(magnitudes):
            top_word = max(zip(sv.data, magnitudes), key=lambda x: x[1])[0].strip()
        else:
            top_word = ""

        # Word-level contributions (first class dimension)
        contrib_scores = sv.values[:, 0] if sv.values.ndim > 1 else sv.values
        contributions  = [
            (w.strip(), round(float(s), 4))
            for w, s in zip(sv.data, contrib_scores)
        ]

        results.append({
            "pair_id":            row["pair_id"],
            "shap_score_pct":     pct,
            "shap_top_word":      top_word,
            "shap_contributions": contributions,
        })

    shap_df = pd.DataFrame(results)
    print(f"  ✓ SHAP done. Avg={shap_df['shap_score_pct'].mean():.1f}% "
          f"Max={shap_df['shap_score_pct'].max():.1f}%")
    return shap_df


# ──────────────────────────────────────────────
# STEP 4 — LIME SCORING
# ──────────────────────────────────────────────

def run_lime(df, pipeline):
    print("  Running LIME (may take ~1 min)...")
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=[str(c) for c in pipeline.classes_]
    )

    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            exp      = explainer.explain_instance(
                row["requirement"], pipeline.predict_proba, num_features=6
            )
            items    = exp.as_list()
            scores   = [abs(s) for _, s in items]
            raw_conf = max(scores) if scores else 0.0
            pct      = round(min(raw_conf / 0.2, 1.0) * 100, 2)
            top_word = max(items, key=lambda x: abs(x[1]))[0].strip() if items else ""
            contribs = [(w.strip(), round(s, 4)) for w, s in items]
        except Exception as e:
            pct, top_word, contribs = 0.0, "", []

        results.append({
            "pair_id":            row["pair_id"],
            "lime_score_pct":     pct,
            "lime_top_word":      top_word,
            "lime_contributions": contribs,
        })

        # Progress every 20 rules
        if (idx + 1) % 20 == 0:
            print(f"    LIME: {idx+1}/{len(df)} done...")

    lime_df = pd.DataFrame(results)
    print(f"  ✓ LIME done. Avg={lime_df['lime_score_pct'].mean():.1f}% "
          f"Max={lime_df['lime_score_pct'].max():.1f}%")
    return lime_df


# ──────────────────────────────────────────────
# STEP 5 — VALIDATOR SCORE (Person 3)
# P3 weighted_risk is 0–1, convert to 0–100
# ──────────────────────────────────────────────

def validator_score_pct(p3_risk: float) -> float:
    return round(min(p3_risk * 100, 100.0), 2)


# ──────────────────────────────────────────────
# STEP 6 — MODEL CONFIDENCE PENALTY
# ──────────────────────────────────────────────

def model_confidence_penalty(pipeline, requirement: str,
                               predicted_label: str) -> float:
    proba = max(pipeline.predict_proba([requirement])[0])
    if predicted_label == "correct":
        return round((1.0 - proba) * 100, 2)     # confident correct = low penalty
    elif predicted_label == "dangerous":
        return round(proba * 100, 2)              # confident dangerous = high penalty
    else:
        return round(proba * 50, 2)               # hallucinated = medium


# ──────────────────────────────────────────────
# STEP 7 — RISK LEVEL
# ──────────────────────────────────────────────

def assign_risk_level(final_score: float, predicted_label: str) -> str:
    if predicted_label == "dangerous":
        return "REJECT"
    if final_score >= THRESHOLD_REJECT:
        return "REJECT"
    elif final_score >= THRESHOLD_REVIEW:
        return "REVIEW"
    return "SAFE"


# ──────────────────────────────────────────────
# STEP 8 — ENSEMBLE DISAGREEMENT
# ──────────────────────────────────────────────

def check_disagreement(shap_pct: float, lime_pct: float) -> str:
    gap = abs(shap_pct - lime_pct)
    if gap > 40:
        return f"HIGH_DISAGREEMENT (gap={gap:.1f}%)"
    elif gap > 20:
        return f"MODERATE_DISAGREEMENT (gap={gap:.1f}%)"
    return "AGREEMENT"


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_risk_engine_v2():
    print(f"\n{'='*72}")
    print(f"  TrustGuard — Risk Engine v2 (Unified)")
    print(f"  Weights: SHAP={WEIGHTS['shap']} | LIME={WEIGHTS['lime']} | "
          f"Validator={WEIGHTS['validator']} | ModelConf={WEIGHTS['model_conf']}")
    print(f"  Thresholds: SAFE < {THRESHOLD_REVIEW} | "
          f"REVIEW {THRESHOLD_REVIEW}–{THRESHOLD_REJECT} | REJECT ≥ {THRESHOLD_REJECT}")
    print(f"{'='*72}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Step 1 — Loading data...")
    df, _ = load_data()
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Step 2 — Training model...")
    pipeline = train_model(df)
    print()

    # ── XAI ──────────────────────────────────────────────────────────────────
    print("Step 3 — Running explainability modules...")
    shap_df = run_shap(df, pipeline)
    lime_df = run_lime(df, pipeline)
    print()

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = df.merge(shap_df[["pair_id","shap_score_pct",
                             "shap_top_word","shap_contributions"]], on="pair_id")
    df = df.merge(lime_df[["pair_id","lime_score_pct",
                             "lime_top_word","lime_contributions"]], on="pair_id")

    # ── Score ─────────────────────────────────────────────────────────────────
    print("Step 4 — Computing final risk scores...")
    results = []

    for _, row in df.iterrows():
        req        = row["requirement"]
        predicted  = pipeline.predict([req])[0]
        model_prob = round(max(pipeline.predict_proba([req])[0]), 4)

        shap_pct   = row["shap_score_pct"]
        lime_pct   = row["lime_score_pct"]
        val_pct    = validator_score_pct(row["p3_weighted_risk"])
        conf_pen   = model_confidence_penalty(pipeline, req, predicted)
        disagreement = check_disagreement(shap_pct, lime_pct)

        final_score = round(
            WEIGHTS["shap"]       * shap_pct
          + WEIGHTS["lime"]       * lime_pct
          + WEIGHTS["validator"]  * val_pct
          + WEIGHTS["model_conf"] * conf_pen,
            2
        )

        risk_level = assign_risk_level(final_score, predicted)

        results.append({
            "Pair ID":              row["pair_id"],
            "Requirement":          req[:80],
            "Actual Label":         row["label"],
            "Predicted Label":      predicted,
            "Correct":              row["label"] == predicted,
            "Model Probability":    model_prob,
            "SHAP Score %":         shap_pct,
            "LIME Score %":         lime_pct,
            "Validator Score %":    val_pct,
            "Model Conf Penalty %": conf_pen,
            "P3 Syntax %":          round(row["p3_syntax_score"]     * 100, 1),
            "P3 Semantic %":        round(row["p3_semantic_score"]    * 100, 1),
            "P3 Compliance %":      round(row["p3_compliance_score"]  * 100, 1),
            "P3 Edge Case %":       round(row["p3_edge_score"]        * 100, 1),
            "SHAP Top Word":        row["shap_top_word"],
            "LIME Top Word":        row["lime_top_word"],
            "SHAP Contributions":   str(row["shap_contributions"][:3]),
            "LIME Contributions":   str(row["lime_contributions"][:3]),
            "XAI Disagreement":     disagreement,
            "Final Score %":        final_score,
            "Risk Level":           risk_level,
            "Hallucination Type":   row["hallucination_type"],
            "P3 Halluc Types":      row["p3_halluc_types"],
            "Security Impact":      row["security_impact"],
            "Compliance Violation": row["compliance_violation"],
        })

    results_df = pd.DataFrame(results)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  TRUSTGUARD RISK ENGINE v2 — FULL RESULTS")
    print(f"{'='*72}")
    print(results_df[[
        "Pair ID", "Actual Label", "Predicted Label",
        "SHAP Score %", "LIME Score %", "Validator Score %",
        "Final Score %", "Risk Level"
    ]].to_string(index=False))

    # ── Summary ───────────────────────────────────────────────────────────────
    accuracy = results_df["Correct"].mean() * 100

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    print(f"  Total Rules:          {len(results_df)}")
    print(f"  Model Accuracy:       {accuracy:.2f}%")

    print(f"\n  Risk Distribution:")
    for level, cnt in results_df["Risk Level"].value_counts().items():
        pct = cnt / len(results_df) * 100
        icon = {"SAFE":"✅","REVIEW":"⚠️ ","REJECT":"🔴"}.get(level,"")
        print(f"    {icon} {level:<8} {cnt:>3}  ({pct:.1f}%)")

    print(f"\n  Average Final Score % by Actual Label:")
    print(f"  (paper metric — want: correct < hallucinated < dangerous)")
    for lbl in ["correct","hallucinated","dangerous"]:
        avg = results_df[results_df["Actual Label"]==lbl]["Final Score %"].mean()
        print(f"    {lbl:<15} {avg:.2f}%")

    print(f"\n  Average component scores by label:")
    print(f"  {'Label':<15} {'SHAP':>7} {'LIME':>7} "
          f"{'Validator':>10} {'Final':>7}")
    print(f"  {'-'*50}")
    for lbl in ["correct","hallucinated","dangerous"]:
        sub = results_df[results_df["Actual Label"]==lbl]
        print(f"  {lbl:<15} "
              f"{sub['SHAP Score %'].mean():>7.1f} "
              f"{sub['LIME Score %'].mean():>7.1f} "
              f"{sub['Validator Score %'].mean():>10.1f} "
              f"{sub['Final Score %'].mean():>7.1f}")

    print(f"\n  Label vs Risk Level (confusion matrix):")
    print(pd.crosstab(results_df["Actual Label"],
                      results_df["Risk Level"]).to_string())

    print(f"\n  XAI Ensemble Disagreement:")
    for status, cnt in results_df["XAI Disagreement"].value_counts().items():
        print(f"    {status:<45} {cnt}")

    mis = results_df[results_df["Correct"]==False]
    print(f"\n  Misclassified Rules ({len(mis)}):")
    if len(mis):
        print(mis[["Pair ID","Actual Label","Predicted Label",
                   "Final Score %","Risk Level"]].to_string(index=False))
    else:
        print("    None!")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)

    json_out = {
        "metadata": {
            "created_at":      datetime.now(UTC).isoformat(),
            "total":           len(results),
            "weights":         WEIGHTS,
            "thresholds":      {
                "review": THRESHOLD_REVIEW,
                "reject": THRESHOLD_REJECT
            },
            "model_accuracy":  f"{accuracy:.2f}%",
            "risk_distribution": results_df["Risk Level"].value_counts().to_dict(),
            "avg_score_by_label": {
                lbl: round(
                    results_df[results_df["Actual Label"]==lbl]["Final Score %"].mean(),
                    2
                )
                for lbl in ["correct","hallucinated","dangerous"]
            },
            "avg_component_by_label": {
                lbl: {
                    "shap":      round(results_df[results_df["Actual Label"]==lbl]["SHAP Score %"].mean(),2),
                    "lime":      round(results_df[results_df["Actual Label"]==lbl]["LIME Score %"].mean(),2),
                    "validator": round(results_df[results_df["Actual Label"]==lbl]["Validator Score %"].mean(),2),
                }
                for lbl in ["correct","hallucinated","dangerous"]
            }
        },
        "results": [
            {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
             for k, v in r.items()}
            for r in results
        ]
    }

    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"\n  ✓ CSV  → {OUT_CSV}")
    print(f"  ✓ JSON → {OUT_JSON}")
    print(f"\n  Risk Engine v2 Complete ✓\n")

    return results_df


if __name__ == "__main__":
    run_risk_engine_v2()