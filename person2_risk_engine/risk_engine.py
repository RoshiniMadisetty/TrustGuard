
# Cell 3 - Full Risk Engine with JSON + Final Score
# ==================================================
# risk_engine.ipynb
# Risk Scoring Engine v1 — TrustGuard Person 2
# Inputs: JSON from Person 1
# Output: LIME% + SHAP% + Verifier% → Final Score%
# ==================================================

import json
import lime
import lime.lime_text
import shap
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==================================================
# STEP 1 — LOAD JSON
# ==================================================

with open("person1_llm_pipeline/data/week4_final_dataset.json", "r") as f:
    raw = json.load(f)

records = []
for pair in raw['pairs']:
    generated_rule = pair.get('generated_rule') or {}
    records.append({
        "pair_id"           : pair['pair_id'],
        "requirement"       : pair['requirement'],
        "label"             : pair['label'],
        "hallucination_type": pair['hallucination_type'],
        "label_confidence"  : pair['label_confidence'],
        "security_impact"   : pair.get('security_impact', 'None'),
        "compliance_violation": ", ".join(pair.get('compliance_violation', [])),
        "parse_success"     : pair['generation_metadata']['parse_success'],
        "action"            : generated_rule.get('action', 'unknown'),
        "protocol"          : generated_rule.get('protocol', 'unknown'),
        "source"            : generated_rule.get('source', 'unknown'),
        "destination"       : generated_rule.get('destination', 'unknown'),
        "destination_port"  : str(generated_rule.get('destination_port', 'unknown')),
    })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} rules from JSON")
print("Label Distribution:\n", df['label'].value_counts())

# ==================================================
# STEP 2 — TRAIN MODEL
# ==================================================

X = df['requirement']
y = df['label']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X, y)
print("\nModel trained. Classes:", pipeline.classes_)

# ==================================================
# STEP 3 — LIME SETUP
# ==================================================

lime_explainer = lime.lime_text.LimeTextExplainer(
    class_names=[str(c) for c in pipeline.classes_]
)

def get_lime_score(rule):
    exp = lime_explainer.explain_instance(
        rule, pipeline.predict_proba, num_features=6
    )
    scores = [abs(s) for _, s in exp.as_list()]
    top_word = max(exp.as_list(), key=lambda x: abs(x[1]))[0].strip()
    raw_conf = max(scores)
    # Normalize to 0-100% (cap at 0.2 as max observed)
    pct = round(min(raw_conf / 0.2, 1.0) * 100, 2)
    return pct, top_word

# ==================================================
# STEP 4 — SHAP SETUP
# ==================================================

shap_explainer = shap.Explainer(
    pipeline.predict_proba,
    shap.maskers.Text(r"\W+")
)

def get_shap_score(shap_val):
    if shap_val.values.ndim > 1:
        values = np.abs(shap_val.values).sum(axis=-1)
    else:
        values = np.abs(shap_val.values)
    top_word = max(zip(shap_val.data, values), key=lambda x: x[1])[0].strip()
    raw_conf = float(max(values))
    # Normalize to 0-100% (cap at 0.4 as max observed)
    pct = round(min(raw_conf / 0.4, 1.0) * 100, 2)
    return pct, top_word

# ==================================================
# STEP 5 — VERIFIER SCORE
# Rules from JSON structure — flag issues in the rule
# ==================================================

def get_verifier_score(row):
    """
    Checks the generated rule fields for red flags.
    Returns a risk score 0-100% based on rule properties.
    Higher % = more suspicious/risky.
    """
    score = 0
    flags = []

    # Parse failed → very risky
    if not row['parse_success']:
        score += 40
        flags.append("parse_failed")

    # Over-permissive: source/dest/protocol all 'any'
    any_count = sum([
        str(row['source']).lower() == 'any',
        str(row['destination']).lower() == 'any',
        str(row['protocol']).lower() == 'any',
        str(row['destination_port']).lower() == 'any'
    ])
    if any_count >= 3:
        score += 40
        flags.append("over_permissive")
    elif any_count == 2:
        score += 20
        flags.append("partially_permissive")

    # Destination is 0.0.0.0/0 (internet-wide)
    if '0.0.0.0/0' in str(row['destination']):
        score += 20
        flags.append("internet_wide_dest")

    # Compliance violation present
    if row['compliance_violation']:
        score += 10
        flags.append("compliance_violation")

    score = min(score, 100)
    return round(score, 2), ", ".join(flags) if flags else "none"

# ==================================================
# STEP 6 — RISK LEVEL FROM FINAL SCORE
# ==================================================

def assign_risk_level(final_score, predicted_label):
    if predicted_label == 'dangerous':
        return 'REJECT'
    if final_score >= 60:
        return 'REJECT'
    elif final_score >= 30:
        return 'REVIEW'
    else:
        return 'SAFE'

# ==================================================
# STEP 7 — RUN ON ALL RULES
# ==================================================

print("\nRunning SHAP on all rules (this may take a minute)...")
all_rules = df['requirement'].tolist()
shap_values = shap_explainer(all_rules)

results = []

for i, (_, row) in enumerate(df.iterrows()):
    rule = row['requirement']

    # Prediction
    prediction = pipeline.predict([rule])[0]
    model_proba = round(max(pipeline.predict_proba([rule])[0]), 4)

    # LIME score
    lime_pct, lime_top = get_lime_score(rule)

    # SHAP score
    shap_pct, shap_top = get_shap_score(shap_values[i])

    # Verifier score
    verifier_pct, verifier_flags = get_verifier_score(row)

    # Final combined score
    final_score = round((lime_pct + shap_pct + verifier_pct) / 3, 2)

    # Risk level
    risk = assign_risk_level(final_score, prediction)

    results.append({
        "Pair ID"            : row['pair_id'],
        "Requirement"        : rule,
        "Actual Label"       : row['label'],
        "Predicted Label"    : prediction,
        "Correct"            : row['label'] == prediction,
        "Model Probability"  : model_proba,
        "LIME Score %"       : lime_pct,
        "SHAP Score %"       : shap_pct,
        "Verifier Score %"   : verifier_pct,
        "Final Score %"      : final_score,
        "LIME Top Word"      : lime_top,
        "SHAP Top Word"      : shap_top,
        "Verifier Flags"     : verifier_flags,
        "Hallucination Type" : row['hallucination_type'],
        "Security Impact"    : row['security_impact'],
        "Risk Level"         : risk
    })

results_df = pd.DataFrame(results)

# ==================================================
# STEP 8 — PRINT RESULTS
# ==================================================

print("\n" + "="*70)
print("TRUSTGUARD RISK ENGINE — FULL RESULTS")
print("="*70)
print(results_df[[
    'Pair ID', 'Requirement', 'Actual Label', 'Predicted Label',
    'LIME Score %', 'SHAP Score %', 'Verifier Score %',
    'Final Score %', 'Risk Level'
]].to_string(index=False))

# ==================================================
# STEP 9 — SUMMARY
# ==================================================

accuracy = results_df['Correct'].mean() * 100
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total Rules          : {len(results_df)}")
print(f"Overall Accuracy     : {accuracy:.2f}%")
print(f"\nRisk Distribution:")
print(results_df['Risk Level'].value_counts().to_string())
print(f"\nAverage Final Score by Label:")
for label in ['correct', 'hallucinated', 'dangerous']:
    avg = results_df[results_df['Actual Label'] == label]['Final Score %'].mean()
    print(f"  {label:15}: {avg:.2f}%")
print(f"\nLabel vs Risk Level:")
print(pd.crosstab(results_df['Actual Label'], results_df['Risk Level']).to_string())
print(f"\nMisclassified Rules:")
mis = results_df[results_df['Correct'] == False]
print(mis[['Pair ID', 'Actual Label', 'Predicted Label', 'Final Score %', 'Risk Level']].to_string(index=False))

# ==================================================
# STEP 10 — SAVE
# ==================================================

results_df.to_csv("person2_risk_engine/risk_engine_v1_results.csv", index=False)
print("\nResults saved to person2_risk_engine/risk_engine_v1_results.csv")
print("\nRisk Engine v1 Complete ✓")