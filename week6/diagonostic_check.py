"""
TrustGuard Week 6 - Diagnostics & Output Exporter v3 (FINAL)
=============================================================
All JSON paths confirmed from debug output.

File → path → metric:
  week6_final_report.json
    key_results.baseline_comparison.trustguard_f1     → f1 (0.9231)
    key_results.baseline_comparison.raw_llm_f1        → baseline
    key_results.baseline_comparison.improvement       → improvement
    key_results.decision_layer.strict_f1              → strict_f1
    key_results.decision_layer.lenient_f1             → lenient_f1
    key_results.decision_layer.safe/review/reject     → counts
    key_results.adversarial.detection_rate            → adversarial rate
    key_results.adversarial.total                     → adversarial total
    dataset.total_records / labelled_records etc      → dataset info

  week6_benchmark_report.json
    binary_classification.auc_roc                     → AUC (0.9658)
    binary_classification.recall                      → recall (0.9677)
    binary_classification.precision                   → precision (0.8824)
    binary_classification.f1_score                    → f1 cross-check
    benchmark_run.*                                   → dataset counts

  week6_xai_disagreement.json
    summary.mean_agreement                            → agreement
    summary.disagreement_count / strong / partial     → XAI stats
    per_record_analysis                               → XAI CSV export

  week6_ensemble_confidence.json
    summary.mean_ensemble                             → mean conf
    summary.high/moderate/low_confidence_count        → tier counts
    records                                           → ensemble CSV

  week6_decisions.json
    summary.safe/review/reject_count                  → decision counts
    summary.override_count                            → overrides
    summary.thresholds_used.*                         → thresholds
    decisions                                         → decisions CSV

Usage:
    python diagnostics_v3.py
    python diagnostics_v3.py --out_dir C:\\Users\\Roshini\\Downloads\\results
    python diagnostics_v3.py --debug
"""

import os, sys, json, csv, argparse, datetime
from pathlib import Path

DEFAULT_WEEK6_DIR = os.path.dirname(os.path.abspath(__file__))

# All 6 files now (added benchmark)
EXPECTED_FILES = {
    "final_report": "week6_final_report.json",
    "benchmark":    "week6_benchmark_report.json",
    "decisions":    "week6_decisions.json",
    "xai":          "week6_xai_disagreement.json",
    "ensemble":     "week6_ensemble_confidence.json",
    "thresholds":   "week6_calibrated_thresholds.json",
}

HEALTH = {
    "min_f1":                   0.85,
    "min_auc":                  0.90,
    "min_recall":               0.90,
    "max_xai_disagreement_pct": 15.0,
    "min_adversarial_rate":     1.0,
    "max_surrogate_r2_std":     0.30,
    "min_surrogate_r2":         0.70,
    "min_high_conf_records":    20,
    "strict_f1_warn":           0.60,
}

# ─── HELPERS ────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all(week6_dir):
    data, missing = {}, []
    for key, fname in EXPECTED_FILES.items():
        fpath = Path(week6_dir) / fname
        if fpath.exists():
            data[key] = load_json(fpath)
        else:
            missing.append(fname)
            data[key] = None
    return data, missing

def dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur

def pct(num, denom):
    return round(100.0 * num / denom, 2) if denom else 0.0

# ─── EXTRACT (all paths now confirmed) ──────────────────────────────────────

def extract_metrics(data):
    m = {}

    rpt = data.get("final_report") or {}
    bm  = data.get("benchmark")    or {}
    dec = data.get("decisions")    or {}
    xai = data.get("xai")          or {}
    ens = data.get("ensemble")     or {}
    thr = data.get("thresholds")   or {}

    kr  = rpt.get("key_results", {})
    dl  = kr.get("decision_layer", {})
    adv = kr.get("adversarial", {})
    bc  = kr.get("baseline_comparison", {})
    ds  = rpt.get("dataset", {})

    # ── Core metrics (benchmark file is authoritative for AUC/recall) ──
    bmc = bm.get("binary_classification", {})
    bmr = bm.get("benchmark_run", {})

    m["f1"]        = bmc.get("f1_score",  bc.get("trustguard_f1", 0))
    m["auc"]       = bmc.get("auc_roc",   0)
    m["recall"]    = bmc.get("recall",    0)
    m["precision"] = bmc.get("precision", 0)
    m["auc_pr"]    = bmc.get("auc_pr",    None)

    # cross-check: baseline_comparison also has trustguard_f1
    m["f1_crosscheck"]       = bc.get("trustguard_f1", None)
    m["baseline_llm_f1"]     = bc.get("raw_llm_f1", 0)
    m["baseline_improvement"]= bc.get("improvement", None)

    # decision layer
    m["strict_f1"]  = dl.get("strict_f1",  None)
    m["lenient_f1"] = dl.get("lenient_f1", None)

    # adversarial
    m["adversarial_rate"]  = adv.get("detection_rate", 0)
    m["adversarial_total"] = adv.get("total", 0)

    # ── Dataset counts ──
    m["total_records"] = ds.get("total_records",    bmr.get("total_records",    215))
    m["labelled"]      = ds.get("labelled_records", bmr.get("labelled_records", 65))
    m["unlabelled"]    = ds.get("synthetic_records", 0)
    m["hallucinated"]  = bmr.get("hallucinated", dl.get("reject_count", 31))
    m["correct"]       = bmr.get("clean",        dl.get("safe_count",   34))

    # ── Decision counts (from decisions.summary — confirmed path) ──
    dec_sum = dec.get("summary", {})
    m["safe_count"]   = dec_sum.get("safe_count",   dl.get("safe_count",   None))
    m["review_count"] = dec_sum.get("review_count", dl.get("review_count", None))
    m["reject_count"] = dec_sum.get("reject_count", dl.get("reject_count", None))
    m["overrides"]    = dec_sum.get("override_count", None)

    # thresholds from decisions.summary.thresholds_used
    thr_used = dec_sum.get("thresholds_used", {})
    pt = thr.get("primary_thresholds", {})
    m["thresh_safe"]   = thr_used.get("safe",   pt.get("safe_threshold",   "N/A"))
    m["thresh_review"] = thr_used.get("review", pt.get("review_threshold", "N/A"))
    m["thresh_reject"] = thr_used.get("reject", "N/A")

    # decisions list for CSV
    dec_list = dec.get("decisions", [])
    if isinstance(dec_list, dict):
        dec_list = [{"record_id": k, **v} for k, v in dec_list.items()]
    m["decisions_list"] = dec_list if isinstance(dec_list, list) else []

    # ── XAI (confirmed: summary.mean_agreement etc) ──
    xs = xai.get("summary", {})
    m["xai_mean_agreement"]     = xs.get("mean_agreement", xs.get("mean_jaccard", 0))
    m["xai_disagreement_count"] = xs.get("disagreement_count", 0)
    m["xai_total"]              = xs.get("n_records", 215)
    m["xai_disagreement_pct"]   = pct(m["xai_disagreement_count"], m["xai_total"])
    m["xai_strong_agree"]       = xs.get("strong_agreement_count", 0)
    m["xai_strong_agree_pct"]   = pct(m["xai_strong_agree"], m["xai_total"])
    m["xai_partial_agree"]      = xs.get("partial_agreement_count", 0)
    m["xai_partial_agree_pct"]  = pct(m["xai_partial_agree"], m["xai_total"])

    # surrogate R² lives in xai_report (week5_xai_report.json) — not loaded here
    # read from xai disagreement config if present
    cfg = xai.get("config", {})
    m["surrogate_r2"]     = cfg.get("surrogate_r2",     None)
    m["surrogate_r2_std"] = cfg.get("surrogate_r2_std", None)
    m["top_shap_feature"] = cfg.get("top_shap_feature", "confidence")
    m["top_shap_value"]   = cfg.get("top_shap_value",   None)
    m["low_signal_features"] = cfg.get("low_signal_features", [])
    m["xai_records"]      = xai.get("per_record_analysis", [])

    # ── Ensemble (confirmed: summary.mean_ensemble) ──
    es = ens.get("summary", {})
    m["mean_ensemble_conf"]  = es.get("mean_ensemble",            0)
    m["high_conf_count"]     = es.get("high_confidence_count",    0)
    m["moderate_conf_count"] = es.get("moderate_confidence_count",0)
    m["low_conf_count"]      = es.get("low_confidence_count",     0)
    m["ensemble_records"]    = ens.get("records", [])

    # ── Per-category breakdown from benchmark records ──
    bm_records = bm.get("records", [])
    cat_counts = {}
    for r in bm_records:
        ht = r.get("hallucination_type", "none")
        if ht and ht != "none":
            cat_counts[ht] = cat_counts.get(ht, 0) + 1
    m["category_counts"] = cat_counts

    return m

# ─── HEALTH CHECKS ───────────────────────────────────────────────────────────

def run_health_checks(m):
    checks = []

    def chk(name, condition, value, threshold, level="FAIL"):
        status = "PASS" if condition else ("WARN" if level == "WARN" else "FAIL")
        checks.append({"check": name, "status": status,
                        "value": value, "threshold": threshold})

    chk("F1 Score",         m["f1"]  >= HEALTH["min_f1"],   round(m["f1"],4),  HEALTH["min_f1"])
    chk("AUC-ROC",          m["auc"] >= HEALTH["min_auc"],  round(m["auc"],4), HEALTH["min_auc"])
    chk("Recall",           m["recall"] >= HEALTH["min_recall"], round(m["recall"],4), HEALTH["min_recall"])
    chk("Adversarial Rate", m["adversarial_rate"] >= HEALTH["min_adversarial_rate"],
        round(m["adversarial_rate"],4), HEALTH["min_adversarial_rate"])

    if m["strict_f1"] is not None:
        chk("Strict F1 (REJECT-only)", m["strict_f1"] >= HEALTH["strict_f1_warn"],
            round(m["strict_f1"],4), HEALTH["strict_f1_warn"], level="WARN")

    labelled = m["labelled"] or 1
    hall = m["hallucinated"] or 0
    ratio = hall / labelled
    chk("Class Balance (hall ratio)", 0.25 <= ratio <= 0.75,
        round(ratio,3), "0.25–0.75", level="WARN")

    dis_pct = m["xai_disagreement_pct"]
    chk("XAI Disagreement %", dis_pct <= HEALTH["max_xai_disagreement_pct"],
        f"{dis_pct}%", f"≤{HEALTH['max_xai_disagreement_pct']}%")

    chk("Mean SHAP-LIME Agreement", m["xai_mean_agreement"] >= 0.50,
        round(m["xai_mean_agreement"],4), "≥0.50")

    if m["surrogate_r2"] is not None:
        chk("Surrogate R²", m["surrogate_r2"] >= HEALTH["min_surrogate_r2"],
            round(m["surrogate_r2"],4), HEALTH["min_surrogate_r2"])
    if m["surrogate_r2_std"] is not None:
        chk("Surrogate R² CV Stability", m["surrogate_r2_std"] <= HEALTH["max_surrogate_r2_std"],
            round(m["surrogate_r2_std"],4), f"≤{HEALTH['max_surrogate_r2_std']}", level="WARN")

    chk("High Confidence Records", m["high_conf_count"] >= HEALTH["min_high_conf_records"],
        m["high_conf_count"], HEALTH["min_high_conf_records"], level="WARN")
    chk("Mean Ensemble Confidence", m["mean_ensemble_conf"] >= 0.65,
        round(m["mean_ensemble_conf"],4), "≥0.65")

    return checks

# ─── CSV EXPORTS ─────────────────────────────────────────────────────────────

def export_decisions_csv(m, out_dir):
    records = m["decisions_list"]
    if not records:
        return None
    out_path = Path(out_dir) / "trustguard_decisions_export.csv"
    keys = ["record_id", "decision", "risk_score", "ensemble_confidence",
            "hallucination_label", "categories_detected", "override_applied", "reason"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in keys}
            if isinstance(row.get("categories_detected"), list):
                row["categories_detected"] = "|".join(row["categories_detected"])
            writer.writerow(row)
    return out_path

def export_xai_csv(m, out_dir):
    records = m["xai_records"]
    if not records:
        return None
    out_path = Path(out_dir) / "trustguard_xai_export.csv"
    priority = ["record_id", "shap_top_feature", "shap_top_value",
                "lime_top_feature", "lime_top_value",
                "agreement_score", "agreement_tier", "label", "decision"]
    all_keys = list(records[0].keys()) if isinstance(records[0], dict) else []
    keys = priority + [k for k in all_keys if k not in priority]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            if isinstance(rec, dict):
                writer.writerow({k: rec.get(k, "") for k in keys})
    return out_path

def export_benchmark_csv(m, bm_data, out_dir):
    """Export per-record benchmark data — useful for per-category analysis."""
    records = bm_data.get("records", []) if bm_data else []
    if not records:
        return None
    out_path = Path(out_dir) / "trustguard_benchmark_records.csv"
    keys = ["record_id", "is_hallucinated", "has_label",
            "risk_score", "hallucination_type"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in keys})
    return out_path

# ─── MARKDOWN ────────────────────────────────────────────────────────────────

def build_markdown(m, checks, missing_files, week6_dir):
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    passes = sum(1 for c in checks if c["status"] == "PASS")
    warns  = sum(1 for c in checks if c["status"] == "WARN")
    fails  = sum(1 for c in checks if c["status"] == "FAIL")
    overall = "🟢 HEALTHY" if fails == 0 and warns <= 2 else \
              ("🟡 WARNINGS" if fails == 0 else "🔴 ISSUES FOUND")

    total = m["total_records"] or 1
    safe_n   = m["safe_count"]   or 0
    review_n = m["review_count"] or 0
    reject_n = m["reject_count"] or 0

    lines = [
        f"# TrustGuard Week 6 — Diagnostics Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Source:** `{week6_dir}`  ",
        f"**Overall:** {overall}",
        f"",
        f"---",
        f"",
        f"## 1. Core Metrics",
        f"",
        f"| Metric | Value | Target | Status |",
        f"|--------|-------|--------|--------|",
        f"| F1 Score (benchmark) | **{m['f1']}** | ≥0.85 | {'✅' if m['f1']>=0.85 else '❌'} |",
        f"| Precision | {m['precision']} | — | — |",
        f"| Recall | {m['recall']} | ≥0.90 | {'✅' if m['recall']>=0.90 else '❌'} |",
        f"| AUC-ROC | {m['auc']} | ≥0.90 | {'✅' if m['auc']>=0.90 else '❌'} |",
        f"| AUC-PR | {m['auc_pr']} | — | — |",
        f"| Strict F1 (REJECT only) | {m['strict_f1']} | ≥0.60 | {'⚠️' if (m['strict_f1'] or 0)<0.60 else '✅'} |",
        f"| Lenient F1 (REVIEW+REJECT) | {m['lenient_f1']} | — | — |",
        f"| Adversarial Detection | {m['adversarial_rate']} ({m['adversarial_total']} prompts) | 1.0 | {'✅' if m['adversarial_rate']>=1.0 else '❌'} |",
        f"| vs Raw LLM Baseline | +{m['baseline_improvement']} | — | ✅ |",
        f"",
        f"## 2. Dataset",
        f"",
        f"| | Count |",
        f"|--|-------|",
        f"| Total records | {m['total_records']} |",
        f"| Labelled | {m['labelled']} |",
        f"| Hallucinated (labelled) | {m['hallucinated']} |",
        f"| Correct (labelled) | {m['correct']} |",
        f"| Unlabelled | {total - (m['labelled'] or 0)} |",
        f"",
        f"## 3. Hallucination Category Breakdown",
        f"",
        f"| Category | Count |",
        f"|----------|-------|",
    ]

    for cat, cnt in sorted(m["category_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {cnt} |")

    lines += [
        f"",
        f"> **Note for paper:** `scope_expansion` and `security_downgrade` are underrepresented.",
        f"> Target 80–100 samples per category for publication-grade per-class metrics.",
        f"",
        f"## 4. Decision Distribution",
        f"",
        f"| Decision | Count | % |",
        f"|----------|-------|----|",
        f"| ✅ SAFE   | {safe_n}   | {pct(safe_n, total)}% |",
        f"| ⚠️  REVIEW | {review_n} | {pct(review_n, total)}% |",
        f"| ❌ REJECT | {reject_n} | {pct(reject_n, total)}% |",
        f"| **Total** | **{total}** | 100% |",
        f"",
        f"Thresholds: SAFE<{m['thresh_safe']} | REVIEW<{m['thresh_review']} | REJECT≥{m['thresh_reject']}  ",
        f"Overrides applied: {m['overrides']}",
        f"",
        f"## 5. XAI Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean SHAP-LIME Agreement | {m['xai_mean_agreement']} |",
        f"| Strong Agreement | {m['xai_strong_agree']} ({m['xai_strong_agree_pct']}%) |",
        f"| Partial Agreement | {m['xai_partial_agree']} ({m['xai_partial_agree_pct']}%) |",
        f"| Disagreement | {m['xai_disagreement_count']} ({m['xai_disagreement_pct']}%) |",
        f"| Surrogate R² | {m['surrogate_r2']} ± {m['surrogate_r2_std']} |",
        f"| Top SHAP Feature | {m['top_shap_feature']} |",
        f"",
        f"## 6. Ensemble Confidence",
        f"",
        f"| Tier | Count |",
        f"|------|-------|",
        f"| High (≥0.75) | {m['high_conf_count']} |",
        f"| Moderate | {m['moderate_conf_count']} |",
        f"| Low | {m['low_conf_count']} |",
        f"| **Mean** | **{m['mean_ensemble_conf']}** |",
        f"",
        f"## 7. Health Checks ({passes}✅ {warns}⚠️ {fails}❌)",
        f"",
        f"| Check | Status | Value | Target |",
        f"|-------|--------|-------|--------|",
    ]

    for c in checks:
        icon = "✅" if c["status"] == "PASS" else ("⚠️" if c["status"] == "WARN" else "❌")
        lines.append(f"| {c['check']} | {icon} {c['status']} | {c['value']} | {c['threshold']} |")

    lines += [
        f"",
        f"## 8. Publication Checklist",
        f"",
        f"| Item | Status |",
        f"|------|--------|",
        f"| F1 ≥ 0.85 | {'✅' if m['f1']>=0.85 else '❌'} |",
        f"| AUC ≥ 0.90 | {'✅' if m['auc']>=0.90 else '❌'} |",
        f"| 100% adversarial detection | {'✅' if m['adversarial_rate']>=1.0 else '❌'} |",
        f"| XAI disagreement < 15% | {'✅' if m['xai_disagreement_pct']<=15 else '❌'} |",
        f"| Labelled n ≥ 300 (target) | {'❌ currently ' + str(m['labelled']) } |",
        f"| All 7 categories covered | {'⚠️ check category table above'} |",
        f"| Strict F1 justified in paper | ⚠️ discuss 3-tier design explicitly |",
        f"| CV variance addressed | ⚠️ R²={m['surrogate_r2']} ±{m['surrogate_r2_std']} — expand dataset |",
    ]

    if missing_files:
        lines += [f"", f"## ⚠️ Missing Files", f""]
        for mf in missing_files:
            lines.append(f"- `{mf}`")

    return "\n".join(lines)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrustGuard Diagnostics v3 (Final)")
    parser.add_argument("--week6_dir", default=DEFAULT_WEEK6_DIR)
    parser.add_argument("--out_dir",   default=None)
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    week6_dir = Path(args.week6_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else week6_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TrustGuard Diagnostics v3 (Final)")
    print("=" * 60)
    print(f"Reading : {week6_dir}")
    print(f"Writing : {out_dir}")
    print()

    data, missing = load_all(week6_dir)
    if missing:
        print(f"⚠️  Missing files: {', '.join(missing)}")
    else:
        print(f"✅ All {len(EXPECTED_FILES)} JSON output files found")

    if args.debug:
        print("\n── DEBUG: top-level keys ──")
        for k, v in data.items():
            if isinstance(v, dict):
                print(f"  {k}: {list(v.keys())}")

    print("\n[1/5] Extracting metrics...")
    m = extract_metrics(data)
    print(f"  F1={m['f1']} | AUC={m['auc']} | Recall={m['recall']} | Precision={m['precision']}")
    print(f"  Adversarial={m['adversarial_rate']} ({m['adversarial_total']} prompts)")
    print(f"  XAI agreement={m['xai_mean_agreement']} | Disagreement={m['xai_disagreement_pct']}%")
    print(f"  Ensemble conf={m['mean_ensemble_conf']} | HighConf={m['high_conf_count']}")
    print(f"  Strict F1={m['strict_f1']} | Lenient F1={m['lenient_f1']}")

    print("\n[2/5] Running health checks...")
    checks = run_health_checks(m)
    passes = sum(1 for c in checks if c["status"] == "PASS")
    warns  = sum(1 for c in checks if c["status"] == "WARN")
    fails  = sum(1 for c in checks if c["status"] == "FAIL")
    print(f"  {passes} PASS | {warns} WARN | {fails} FAIL")
    for c in checks:
        icon = "✅" if c["status"] == "PASS" else ("⚠️ " if c["status"] == "WARN" else "❌")
        print(f"    {icon} {c['check']}: {c['value']} (target: {c['threshold']})")

    print("\n[3/5] Exporting CSVs...")
    dp = export_decisions_csv(m, out_dir)
    xp = export_xai_csv(m, out_dir)
    bp = export_benchmark_csv(m, data.get("benchmark"), out_dir)
    print(f"  Decisions  CSV : {'✅ ' + dp.name if dp else '⚠️  no data'}")
    print(f"  XAI        CSV : {'✅ ' + xp.name if xp else '⚠️  no per_record_analysis'}")
    print(f"  Benchmark  CSV : {'✅ ' + bp.name if bp else '⚠️  no benchmark records'}")

    print("\n[4/5] Saving health checks JSON...")
    health_path = out_dir / "trustguard_health_checks.json"
    with open(health_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {"pass": passes, "warn": warns, "fail": fails},
            "checks": checks,
            "metrics": {k: v for k, v in m.items() if not isinstance(v, list)},
        }, f, indent=2)
    print(f"  ✅ {health_path.name}")

    print("\n[5/5] Building markdown report...")
    md = build_markdown(m, checks, missing, week6_dir)
    md_path = out_dir / "trustguard_diagnostics_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  ✅ {md_path.name}")

    print()
    print("=" * 60)
    overall = "🟢 HEALTHY" if fails == 0 and warns <= 2 else \
              ("🟡 WARNINGS" if fails == 0 else "🔴 ISSUES FOUND")
    print(f"Overall: {overall}")
    print("=" * 60)
    print(f"\n📄 {md_path}")
    print(f"📊 {out_dir / 'trustguard_decisions_export.csv'}")
    print(f"📊 {out_dir / 'trustguard_xai_export.csv'}")
    print(f"📊 {out_dir / 'trustguard_benchmark_records.csv'}")
    print(f"🔍 {health_path}")

if __name__ == "__main__":
    main()