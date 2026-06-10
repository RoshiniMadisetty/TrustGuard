

# Cell 3 - Full Code
# ==========================
# shap_analysis.ipynb
# SHAP Explainability Module
# TrustGuard — Person 2
# ==========================

import json
import shap
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================
# LOAD JSON
# ==========================

with open("person1_llm_pipeline/data/week4_final_dataset.json", "r") as f:
    raw = json.load(f)

records = []
for pair in raw['pairs']:
    records.append({
        "pair_id"           : pair['pair_id'],
        "requirement"       : pair['requirement'],
        "label"             : pair['label'],
        "hallucination_type": pair['hallucination_type'],
        "label_confidence"  : pair['label_confidence'],
    })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} rules from JSON")
print("Label Distribution:\n", df['label'].value_counts())

# ==========================
# TRAIN MODEL
# ==========================

X = df['requirement']
y = df['label']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X, y)
print("\nModel trained. Classes:", pipeline.classes_)

# ==========================
# SHAP FUNCTIONS
# ==========================

def build_shap_explainer(predict_fn):
    return shap.Explainer(predict_fn, shap.maskers.Text(r"\W+"))

def get_shap_confidence(shap_val):
    if shap_val.values.ndim > 1:
        values = np.abs(shap_val.values).sum(axis=-1)
    else:
        values = np.abs(shap_val.values)
    return round(float(max(values)), 4)

def get_shap_score_pct(shap_val):
    raw = get_shap_confidence(shap_val)
    return round(min(raw / 0.4, 1.0) * 100, 2)

def get_shap_top_word(shap_val):
    if shap_val.values.ndim > 1:
        scores = np.abs(shap_val.values).sum(axis=-1)
    else:
        scores = np.abs(shap_val.values)
    return max(zip(shap_val.data, scores), key=lambda x: x[1])[0].strip()

def get_shap_word_contributions(shap_val):
    if shap_val.values.ndim > 1:
        scores = shap_val.values[:, 0]
    else:
        scores = shap_val.values
    return [(w.strip(), round(float(s), 4)) for w, s in zip(shap_val.data, scores)]

def run_shap_on_dataset(df, pipeline):
    explainer = build_shap_explainer(pipeline.predict_proba)
    rules = df['requirement'].tolist()
    shap_values = explainer(rules)
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "pair_id"            : row['pair_id'],
            "requirement"        : row['requirement'],
            "actual_label"       : row['label'],
            "shap_confidence"    : get_shap_confidence(shap_values[i]),
            "shap_score_pct"     : get_shap_score_pct(shap_values[i]),
            "shap_top_word"      : get_shap_top_word(shap_values[i]),
            "shap_contributions" : get_shap_word_contributions(shap_values[i])
        })
    return pd.DataFrame(results)

# ==========================
# RUN SHAP
# ==========================

print("\nRunning SHAP on all rules (may take a minute)...")
shap_df = run_shap_on_dataset(df, pipeline)

print("\n==============================")
print("SHAP ANALYSIS RESULTS")
print("==============================")
print(shap_df[[
    'pair_id', 'requirement', 'actual_label',
    'shap_confidence', 'shap_score_pct', 'shap_top_word'
]].to_string(index=False))

print("\nAverage SHAP Score % by Label:")
for label in ['correct', 'hallucinated', 'dangerous']:
    avg = shap_df[shap_df['actual_label'] == label]['shap_score_pct'].mean()
    print(f"  {label:15}: {avg:.2f}%")

print("\nWord Contributions for First 3 Rules:")
for _, row in shap_df.head(3).iterrows():
    print(f"\n{row['pair_id']}: {row['requirement']}")
    print(f"Contributions: {row['shap_contributions']}")

shap_df.to_csv("person2_risk_engine/shap_results.csv", index=False)
print("\nSHAP Analysis Complete ✓")