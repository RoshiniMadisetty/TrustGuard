

# Cell 3 - Full Code
# ==========================
# lime_analysis.ipynb
# LIME Explainability Module
# TrustGuard — Person 2
# ==========================

import json
import lime
import lime.lime_text
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
# LIME FUNCTIONS
# ==========================

def build_lime_explainer(class_names):
    return lime.lime_text.LimeTextExplainer(
        class_names=[str(c) for c in class_names]
    )

def get_lime_explanation(explainer, rule, predict_fn, num_features=6):
    return explainer.explain_instance(
        rule, predict_fn, num_features=num_features
    )

def get_lime_confidence(explanation):
    scores = [abs(score) for _, score in explanation.as_list()]
    return round(max(scores), 4)

def get_lime_score_pct(explanation):
    raw = get_lime_confidence(explanation)
    return round(min(raw / 0.2, 1.0) * 100, 2)

def get_lime_top_word(explanation):
    return max(explanation.as_list(), key=lambda x: abs(x[1]))[0].strip()

def get_lime_word_contributions(explanation):
    return [(w.strip(), round(s, 4)) for w, s in explanation.as_list()]

def run_lime_on_dataset(df, pipeline, num_features=6):
    explainer = build_lime_explainer(pipeline.classes_)
    results = []
    for _, row in df.iterrows():
        rule = row['requirement']
        exp = get_lime_explanation(explainer, rule, pipeline.predict_proba, num_features)
        results.append({
            "pair_id"           : row['pair_id'],
            "requirement"       : rule,
            "actual_label"      : row['label'],
            "lime_confidence"   : get_lime_confidence(exp),
            "lime_score_pct"    : get_lime_score_pct(exp),
            "lime_top_word"     : get_lime_top_word(exp),
            "lime_contributions": get_lime_word_contributions(exp)
        })
    return pd.DataFrame(results)

# ==========================
# RUN LIME
# ==========================

lime_df = run_lime_on_dataset(df, pipeline)

print("\n==============================")
print("LIME ANALYSIS RESULTS")
print("==============================")
print(lime_df[[
    'pair_id', 'requirement', 'actual_label',
    'lime_confidence', 'lime_score_pct', 'lime_top_word'
]].to_string(index=False))

print("\nAverage LIME Score % by Label:")
for label in ['correct', 'hallucinated', 'dangerous']:
    avg = lime_df[lime_df['actual_label'] == label]['lime_score_pct'].mean()
    print(f"  {label:15}: {avg:.2f}%")

lime_df.to_csv("person2_risk_engine/lime_results.csv", index=False)
print("\nLIME Analysis Complete ✓")