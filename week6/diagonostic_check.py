"""
TrustGuard - XAI Disagreement Diagnostic
Run this from your week6 folder:
    python diagnose_xai.py

It reads week5_xai_report.json and week6_xai_disagreement.json
and tells us exactly WHY disagreement is 87.9%.
"""

import json
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent

print("=" * 60)
print("TrustGuard XAI Disagreement Diagnostic")
print("=" * 60)

# ── 1. Read the XAI report ────────────────────────────────────────
xai_path = WORK_DIR / "week5_xai_report.json"
if not xai_path.exists():
    print(f"ERROR: {xai_path} not found")
    exit(1)

with open(xai_path) as f:
    xai = json.load(f)

print("\n[1] XAI Report Summary")
run = xai.get("xai_run", {})
print(f"    n_samples       : {run.get('n_samples')}")
print(f"    n_features      : {run.get('n_features')}")
print(f"    feature_names   : {run.get('feature_names')}")
print(f"    surrogate R²    : {run.get('surrogate_r2_cv_mean')} ± {run.get('surrogate_r2_cv_std')}")
print(f"    lime_samples    : {run.get('lime_samples')}")

shap_imp = xai.get("shap", {}).get("global_feature_importance", {})
print(f"\n[2] Top 5 SHAP features:")
for i, (k, v) in enumerate(list(shap_imp.items())[:5]):
    print(f"    {i+1}. {k}: {v:.4f}")

lime_data = xai.get("lime", {})
print(f"\n[3] LIME samples present: {list(lime_data.keys())}")
for k, v in lime_data.items():
    weights = v.get("lime_weights", {})
    top_lime = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"    {k}: top features = {[f[0] for f in top_lime]}")

# ── 2. Read disagreement report ───────────────────────────────────
dis_path = WORK_DIR / "week6_xai_disagreement.json"
if not dis_path.exists():
    print(f"\nERROR: {dis_path} not found - run the pipeline first")
    exit(1)

with open(dis_path) as f:
    dis = json.load(f)

print("\n[4] Disagreement Report Summary")
summary = dis.get("summary", {})
print(f"    n_records        : {summary.get('n_records', summary.get('total_records'))}")
print(f"    mean_agreement   : {summary.get('mean_agreement')}")
print(f"    std_agreement    : {summary.get('std_agreement')}")
print(f"    strong_agreement : {summary.get('strong_agreement_count')} ({summary.get('strong_agreement_pct')}%)")
print(f"    partial_agreement: {summary.get('partial_agreement_count')} ({summary.get('partial_agreement_pct')}%)")
print(f"    disagreement     : {summary.get('disagreement_count')} ({summary.get('disagreement_pct')}%)")

# ── 3. Look at what the disagreement module actually compares ─────
print("\n[5] Sample disagreement records (first 5):")
records = dis.get("records", [])[:5]
for r in records:
    print(f"    record_id={r.get('record_id')} "
          f"agreement={r.get('agreement_score', r.get('jaccard_similarity', '?')):.3f} "
          f"shap_top={r.get('shap_top_features', r.get('top_shap_features', '?'))[:2] if isinstance(r.get('shap_top_features', r.get('top_shap_features')), list) else '?'} "
          f"lime_top={r.get('lime_top_features', r.get('top_lime_features', '?'))[:2] if isinstance(r.get('lime_top_features', r.get('top_lime_features')), list) else '?'}")

# Print the raw keys so we know the exact field names
if records:
    print(f"\n[6] Raw keys in disagreement record: {list(records[0].keys())}")
    print(f"    Full first record: ")
    for k, v in records[0].items():
        print(f"      {k}: {v}")

# ── 4. Check what the disagreement MODULE actually reads ──────────
print("\n[7] What shap_lime_disagreement module reads from week5_xai_report.json:")
# The module reads SHAP and LIME feature attributions per record
# Let's check if per_record_examples exist
per_rec = xai.get("shap", {}).get("per_record_examples", [])
print(f"    per_record_examples count: {len(per_rec)}")
if per_rec:
    print(f"    First record shap_values keys: {list(per_rec[0].get('shap_values', {}).keys())[:5]}")

# LIME per-record
print(f"    LIME records: {list(lime_data.keys())}")

print("\n[8] KEY QUESTION - feature name overlap between SHAP and LIME:")
shap_features = set(shap_imp.keys())
lime_features = set()
for sample in lime_data.values():
    lime_features.update(sample.get("lime_weights", {}).keys())

print(f"    SHAP features : {sorted(shap_features)[:8]}")
print(f"    LIME features : {sorted(lime_features)[:8]}")
print(f"    Overlap       : {shap_features & lime_features}")
print(f"    SHAP only     : {shap_features - lime_features}")
print(f"    LIME only     : {lime_features - shap_features}")

print("\n" + "=" * 60)
print("Diagnostic complete.")
print("=" * 60)