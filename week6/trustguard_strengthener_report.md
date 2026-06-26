# TrustGuard Framework Strengthening Report
Generated: 2026-06-26 12:31:26

---

## Summary

| Module | Status | Key Output |
|--------|--------|------------|
| Per-Category Metrics | ✅ | 7 categories evaluated |
| Bootstrap CI (500 runs) | ✅ | F1=0.9207 ± 0.0350 [95% CI: 0.8444–0.9851] |
| SHAP Consolidation | ✅ | Top 4 features = 98.8% SHAP |
| Decision Justification | ✅ | Lenient F1=0.9863 framing ready |
| Labelling Assistant | ✅ | ~251 records need human review. 6 can be auto-accepted from suggestions. |

## What To Do Next

1. **Label unlabelled records**: Open `trustguard_labelling_assistant.csv`.
   Fill the `human_label` column. High-confidence rows can be batch-accepted.
   Target: label at least 50 more → benchmark on 115+ records.

2. **Copy paper tables**: `trustguard_paper_tables.md` has all 6 tables
   ready to paste into your IEEE/Springer draft.

3. **Add bootstrap CI to paper**: Report F1 as
   `0.9207 ± 0.035 [95% CI: 0.8444–0.9851]`
   This directly addresses the single-run sensitivity concern.

4. **Use the LaTeX paragraph**: Copy the decision justification LaTeX from
   `trustguard_paper_tables.md` Section 6 directly into your paper.

5. **Dataset expansion plan** (later, with GPU):
   - Generate 300–500 more labelled records via Ollama
   - Target: 80–100 per hallucination category
   - Run orchestrator on expanded dataset → diagnostics → paper update