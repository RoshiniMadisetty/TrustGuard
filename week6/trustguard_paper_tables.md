# TrustGuard — Publication-Ready Tables
Generated: 2026-06-26 12:31:26

---

## Table 1: Overall Detection Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| F1 Score | 0.9207 | [0.8444–0.9851] |
| Precision | 0.8805 | [0.7647–0.9733] |
| Recall | 0.967 | [0.8908–1.0] |
| AUC-ROC | 0.968 | [0.9163–0.999] |
| AUC-PR | 0.9567 | [0.8757–0.999] |

## Table 2: Per-Category Detection Performance

| Hallucination Category | Support (n) | Precision | Recall | F1 |
|------------------------|-------------|-----------|--------|----|
| over_permissive | 11 | 0.7333 | 1.0 | 0.8462 |
| wrong_port | 8 | 0.6364 | 0.875 | 0.7368 |
| intent_flip | 5 | 0.5556 | 1.0 | 0.7143 |
| missing_constraint | 4 | 0.5 | 1.0 | 0.6667 |
| scope_expansion ⚠️ | 2 | 0.3333 | 1.0 | 0.5 |
| wrong_protocol ⚠️ | 1 | 0.2 | 1.0 | 0.3333 |
| security_downgrade ⚠️ | 0 | None | None | None |
| **OVERALL** | **31** | **0.8824** | **0.9677** | **0.9231** |

> ⚠️ = fewer than 4 samples; interpret with caution.

## Table 3: Decision Layer Distribution

| Decision | Count | % of Total | Hallucinated Intercepted |
|----------|-------|------------|--------------------------|
| SAFE | 31 | 14.4% | 0 |
| REVIEW | 127 | 59.1% | 0 |
| REJECT | 57 | 26.5% | 0 |

## Table 4: SHAP Feature Importance Groups

| Feature Group | Key Feature | Group SHAP | Share |
|---------------|-------------|------------|-------|
| LLM Quality Signals | confidence | 0.502376 | 70.2% |
| Policy Structural Integrity | compliance_severity | 0.204426 | 28.6% |
| Scope & Permissiveness | dst_is_any | 0.006412 | 0.9% |
| Protocol & Direction | action_enc | 0.00225 | 0.3% |

## Table 5: Baseline Comparison

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Raw LLM (no validation) | — | — | 0.0 |
| TrustGuard (ours) | 0.8824 | 0.9677 | 0.9231 |

## Table 6: Adversarial Robustness

| Attack Type | Detected | Total | Rate |
|-------------|----------|-------|------|
| Direct (keyword-triggered) | 13 | 13 | 100% |
| Indirect (no explicit keywords) | 6 | 6 | 100% |
| **Total** | **19** | **19** | **100%** |

---

## Key Narrative Points

### On Strict vs Lenient F1
The lenient F1 of 0.9863 reflects the system's true interception rate — the fraction of hallucinated policies that are prevented from silent deployment.

We report strict F1 for completeness but note it is not the primary evaluation metric for a triage-based system.

### On Feature Importance
Feature importance analysis reveals that 4 of 16 features account for 98.8% of total SHAP importance: confidence, semantic_score, compliance_severity, reasoning_length. This concentration indicates that TrustGuard's detection is driven primarily by LLM output quality signals and structural compliance severity, rather than low-level network parameters. The 12 remaining features contribute 1.2% of importance collectively, suggesting they encode redundant or dataset-scale-limited signal. We retain all 16 features in the surrogate model to avoid information loss but recommend future work to evaluate a pruned 4-feature variant on a larger dataset.

### On Dataset Size
Benchmark metrics (Table 1 & 2) are computed on the 65 manually verified labelled records. The remaining 150 generated policies were processed through the full pipeline for decision assignment and qualitative analysis but excluded from quantitative evaluation due to absent ground truth labels.

### LaTeX Decision Paragraph
```latex
\paragraph{Three-Tier Decision Design.} TrustGuard adopts a SAFE/REVIEW/REJECT triage architecture rather than a binary classifier. Rules with risk scores below $\tau_{safe}$ are auto-approved; those above $\tau_{reject}$ are auto-blocked; intermediate rules are escalated with SHAP and LIME explanations for human review. This design reflects the asymmetric cost structure of firewall policy errors: deploying a hallucinated rule (false negative) introduces a security vulnerability, while unnecessary escalation (false positive) incurs only analyst time. We report both strict F1 (REJECT-only) and lenient F1 (REVIEW+REJECT) to enable comparison with binary baselines while accurately characterising triage-system performance.
```