"""
TrustGuard - Diagnostic: XAI Surrogate + LIME Coverage
Run from week6 folder: python diagnose_xai2.py
"""
import json
import numpy as np
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent

print("=" * 60)
print("Diagnostic: Surrogate Quality + LIME Coverage")
print("=" * 60)

# 1. XAI report
xai = json.load(open(WORK_DIR / "week5_xai_report.json"))
run = xai.get("xai_run", {})
print(f"\n[1] Surrogate")
print(f"    n_samples      : {run.get('n_samples')}")
print(f"    feature_names  : {run.get('feature_names')}")
print(f"    R² mean        : {run.get('surrogate_r2_cv_mean')}")
print(f"    R² std         : {run.get('surrogate_r2_cv_std')}")

per_rec = xai.get("shap", {}).get("per_record_examples", [])
print(f"\n[2] SHAP per-record examples: {len(per_rec)}")

lime = xai.get("lime", {})
print(f"\n[3] LIME samples: {len(lime)} ({list(lime.keys())})")
for k, v in lime.items():
    print(f"    {k}: record_id={v.get('record_id')} "
          f"n_weights={len(v.get('lime_weights', {}))}")

# 2. Validation results — labelled vs unlabelled
val = json.load(open(WORK_DIR / "week6_validation_results.json"))
records = val.get("records", [])
labelled   = [r for r in records if r.get("has_label")]
unlabelled = [r for r in records if not r.get("has_label")]
hall = [r for r in labelled if r.get("is_hallucinated") == 1]
corr = [r for r in labelled if r.get("is_hallucinated") == 0]
print(f"\n[4] Dataset split")
print(f"    total      : {len(records)}")
print(f"    labelled   : {len(labelled)} (hall={len(hall)}, correct={len(corr)})")
print(f"    unlabelled : {len(unlabelled)}")

# 3. Risk score distribution for labelled records
hall_scores = [r.get("risk_score", 0) for r in hall]
corr_scores = [r.get("risk_score", 0) for r in corr]
print(f"\n[5] Risk score distribution (labelled)")
print(f"    hallucinated: mean={np.mean(hall_scores):.3f} "
      f"std={np.std(hall_scores):.3f} "
      f"min={np.min(hall_scores):.3f} max={np.max(hall_scores):.3f}")
print(f"    correct:      mean={np.mean(corr_scores):.3f} "
      f"std={np.std(corr_scores):.3f} "
      f"min={np.min(corr_scores):.3f} max={np.max(corr_scores):.3f}")

# 4. Feature separability check
print(f"\n[6] Feature separability (mean values hallucinated vs correct)")
print(f"    {'Feature':<25} {'Hall mean':>10} {'Corr mean':>10} {'Delta':>8}")
feat_names = run.get("feature_names", [])
# We need to reconstruct features — check if edge case scores exist
try:
    edge = json.load(open(WORK_DIR / "week6_edge_case_scores.json"))
    edge_rec = edge.get("records", [])
    edge_map = {r["record_id"]: r for r in edge_rec}
    
    ACTION_MAP    = {"ALLOW": 0, "DENY": 1, "DROP": 2}
    PROTOCOL_MAP  = {"TCP": 0, "UDP": 1, "ICMP": 2, "ANY": 3}
    DIRECTION_MAP = {"INBOUND": 0, "OUTBOUND": 1, "BOTH": 2}
    SEV_MAP       = {"INFO":0,"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4}
    
    def is_any(v):
        return str(v).strip().upper() in ("ANY", "0.0.0.0/0", "ANY/ANY", "*", "")
    
    def extract(r):
        p = r.get("parsed_policy") or {}
        v = r.get("validation") or {}
        try: dst_port = float(p.get("dst_port", -1))
        except: dst_port = -1.0
        return [
            ACTION_MAP.get(p.get("action",""), -1),
            PROTOCOL_MAP.get(p.get("protocol",""), -1),
            DIRECTION_MAP.get(p.get("direction",""), -1),
            1.0 if is_any(p.get("src_ip","")) else 0.0,
            1.0 if is_any(p.get("dst_ip","")) else 0.0,
            1.0 if is_any(p.get("src_port","")) else 0.0,
            1.0 if is_any(p.get("dst_port","")) else 0.0,
            dst_port,
            float(p.get("confidence", 0.5)),
            float(p.get("priority", 500)) / 1000.0,
            1.0 if p.get("reasoning","").count("Step") >= 3 else 0.0,
            min(float(len(p.get("reasoning",""))), 2000.0) / 2000.0,
            1.0 if (v.get("syntax") or {}).get("valid", False) else 0.0,
            float((v.get("semantic") or {}).get("similarity_score", 0.5)),
            float(SEV_MAP.get((v.get("compliance") or {}).get("max_severity","INFO"), 0)),
            float(len((v.get("edge_case") or {}).get("triggered_cases", []))),
        ]
    
    hall_feats = np.array([extract(r) for r in hall])
    corr_feats = np.array([extract(r) for r in corr])
    
    names = ["action_enc","protocol_enc","direction_enc","src_is_any","dst_is_any",
             "src_port_is_any","dst_port_is_any","dst_port_numeric","confidence",
             "priority_norm","has_complete_cot","reasoning_length","syntax_valid",
             "semantic_score","compliance_severity","edge_case_count"]
    
    for i, name in enumerate(names):
        hm = np.mean(hall_feats[:, i])
        cm = np.mean(corr_feats[:, i])
        print(f"    {name:<25} {hm:>10.3f} {cm:>10.3f} {hm-cm:>8.3f}")
except Exception as e:
    print(f"    (skipped: {e})")

# 5. Benchmark report
bm = json.load(open(WORK_DIR / "week6_benchmark_report.json"))
bc = bm.get("binary_classification", {})
print(f"\n[7] Benchmark (detector, threshold=0.10)")
print(f"    F1={bc.get('f1_score')} P={bc.get('precision')} "
      f"R={bc.get('recall')} AUC={bc.get('auc_roc')}")

# 6. Threshold calibration
try:
    thr = json.load(open(WORK_DIR / "week6_calibrated_thresholds.json"))
    print(f"\n[8] Threshold calibration (external module)")
    pt = thr.get("primary_thresholds", {})
    print(f"    safe={pt.get('safe_threshold')} review={pt.get('review_threshold')}")
    ev = thr.get("evaluation", {})
    print(f"    F1={ev.get('f1_score')} P={ev.get('precision')} R={ev.get('recall')}")
except Exception as e:
    print(f"\n[8] Threshold calibration: {e}")

print("\n" + "=" * 60)
print("Diagnostic complete. Paste output back for analysis.")
print("=" * 60)