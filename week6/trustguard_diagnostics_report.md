# TrustGuard Week 6 — Diagnostics Report

**Generated:** 2026-06-24 14:07:06  
**Source:** `C:\Users\Roshini\Downloads\firewall_validator\firewall_validator\week6`  
**Overall:** 🟢 HEALTHY

---

## 1. Core Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| F1 Score (benchmark) | **0.9231** | ≥0.85 | ✅ |
| Precision | 0.8824 | — | — |
| Recall | 0.9677 | ≥0.90 | ✅ |
| AUC-ROC | 0.9658 | ≥0.90 | ✅ |
| AUC-PR | 0.9537 | — | — |
| Strict F1 (REJECT only) | 0.4706 | ≥0.60 | ⚠️ |
| Lenient F1 (REVIEW+REJECT) | 0.9863 | — | — |
| Adversarial Detection | 1.0 (19 prompts) | 1.0 | ✅ |
| vs Raw LLM Baseline | +0.9231 | — | ✅ |

## 2. Dataset

| | Count |
|--|-------|
| Total records | 215 |
| Labelled | 65 |
| Hallucinated (labelled) | 31 |
| Correct (labelled) | 34 |
| Unlabelled | 150 |

## 3. Hallucination Category Breakdown

| Category | Count |
|----------|-------|
| over_permissive | 11 |
| wrong_port | 8 |
| intent_flip | 5 |
| missing_constraint | 4 |
| scope_expansion | 2 |
| wrong_protocol | 1 |

> **Note for paper:** `scope_expansion` and `security_downgrade` are underrepresented.
> Target 80–100 samples per category for publication-grade per-class metrics.

## 4. Decision Distribution

| Decision | Count | % |
|----------|-------|----|
| ✅ SAFE   | 31   | 14.42% |
| ⚠️  REVIEW | 127 | 59.07% |
| ❌ REJECT | 57 | 26.51% |
| **Total** | **215** | 100% |

Thresholds: SAFE<0.01 | REVIEW<0.42174 | REJECT≥N/A  
Overrides applied: 11

## 5. XAI Summary

| Metric | Value |
|--------|-------|
| Mean SHAP-LIME Agreement | 0.534 |
| Strong Agreement | 98 (45.58%) |
| Partial Agreement | 102 (47.44%) |
| Disagreement | 15 (6.98%) |
| Surrogate R² | None ± None |
| Top SHAP Feature | confidence |

## 6. Ensemble Confidence

| Tier | Count |
|------|-------|
| High (≥0.75) | 33 |
| Moderate | 182 |
| Low | 0 |
| **Mean** | **0.7034** |

## 7. Health Checks (9✅ 1⚠️ 0❌)

| Check | Status | Value | Target |
|-------|--------|-------|--------|
| F1 Score | ✅ PASS | 0.9231 | 0.85 |
| AUC-ROC | ✅ PASS | 0.9658 | 0.9 |
| Recall | ✅ PASS | 0.9677 | 0.9 |
| Adversarial Rate | ✅ PASS | 1.0 | 1.0 |
| Strict F1 (REJECT-only) | ⚠️ WARN | 0.4706 | 0.6 |
| Class Balance (hall ratio) | ✅ PASS | 0.477 | 0.25–0.75 |
| XAI Disagreement % | ✅ PASS | 6.98% | ≤15.0% |
| Mean SHAP-LIME Agreement | ✅ PASS | 0.534 | ≥0.50 |
| High Confidence Records | ✅ PASS | 33 | 20 |
| Mean Ensemble Confidence | ✅ PASS | 0.7034 | ≥0.65 |

## 8. Publication Checklist

| Item | Status |
|------|--------|
| F1 ≥ 0.85 | ✅ |
| AUC ≥ 0.90 | ✅ |
| 100% adversarial detection | ✅ |
| XAI disagreement < 15% | ✅ |
| Labelled n ≥ 300 (target) | ❌ currently 65 |
| All 7 categories covered | ⚠️ check category table above |
| Strict F1 justified in paper | ⚠️ discuss 3-tier design explicitly |
| CV variance addressed | ⚠️ R²=None ±None — expand dataset |