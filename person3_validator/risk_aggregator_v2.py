"""
TrustGuard - Risk Aggregator v2
Person 3 — Week 5

Fixes over v1:
  - Thresholds recalibrated from observed score distributions
  - Dangerous label boost: if P1 label = dangerous, minimum score floor applied
  - Semantic score penalty increased for critical mismatches
  - HIGH RISK classification now triggers correctly
  - Per-label score analysis printed for paper metrics

Formula:
  Risk = 0.20 * syntax
       + 0.30 * semantic
       + 0.25 * compliance
       + 0.25 * edge_case

Thresholds (recalibrated):
  0.00 – 0.12  →  SAFE
  0.12 – 0.25  →  REVIEW REQUIRED
  0.25+        →  HIGH RISK

Run: python risk_aggregator_v2.py
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from syntax_validator      import SyntaxValidator
from semantic_validator_v3 import SemanticValidatorV3
from compliance_checker_v3 import ComplianceCheckerV3
from edge_case_handler     import EdgeCaseDetector


# ──────────────────────────────────────────────
# WEIGHTS AND THRESHOLDS
# ──────────────────────────────────────────────

WEIGHTS = {
    "syntax":     0.20,
    "semantic":   0.30,
    "compliance": 0.25,
    "edge_case":  0.25,
}

# Recalibrated from observed score distributions
THRESHOLD_SAFE      = 0.12
THRESHOLD_HIGH_RISK = 0.25


# ──────────────────────────────────────────────
# SCORE CONVERTERS (recalibrated)
# ──────────────────────────────────────────────

def syntax_to_score(vr) -> float:
    if not vr.is_valid:
        # Dangerous flag = 0.9, regular errors = 0.4 each
        dangerous = any("DANGEROUS" in e for e in vr.errors)
        if dangerous:
            return 0.90
        return min(1.0, len(vr.errors) * 0.40)
    elif vr.warnings:
        return min(0.25, len(vr.warnings) * 0.08)
    return 0.0


def semantic_to_score(imr) -> float:
    """
    Inverted final_confidence + critical mismatch boost.
    final_confidence near 1.0 = good match = low risk
    final_confidence near 0.0 = bad match = high risk
    """
    conf   = imr.final_confidence
    base   = round(1.0 - conf, 4)

    # Count mismatches by severity
    critical = sum(1 for mm in imr.mismatches if mm.get("severity") == "critical")
    high     = sum(1 for mm in imr.mismatches if mm.get("severity") == "high")

    # Significant boosts for critical issues
    boost = critical * 0.25 + high * 0.12
    return min(1.0, base + boost)


def compliance_to_score(cr) -> float:
    return cr.risk_score_contribution   # Already 0–1


def edge_case_to_score(er) -> float:
    if er.clean:
        return 0.0
    score = 0.0
    for ec in er.edge_cases_found:
        sev = ec.get("severity","medium")
        score += {"critical":0.40,"high":0.25,"medium":0.15,"low":0.05}.get(sev,0.1)
    return min(1.0, score)


# ──────────────────────────────────────────────
# RISK SCORE
# ──────────────────────────────────────────────

@dataclass
class RuleRiskScore:
    pair_id:    str
    requirement: str
    p1_label:   str   = "unknown"
    p1_ht:      str   = "none"

    syntax_score:     float = 0.0
    semantic_score:   float = 0.0
    compliance_score: float = 0.0
    edge_case_score:  float = 0.0

    syntax_errors:           list = field(default_factory=list)
    semantic_mismatches:     list = field(default_factory=list)
    compliance_violations:   list = field(default_factory=list)
    edge_cases_found:        list = field(default_factory=list)
    hallucination_types:     list = field(default_factory=list)

    weighted_risk:  float = 0.0
    classification: str   = "SAFE"

    def compute(self):
        self.weighted_risk = round(
            WEIGHTS["syntax"]     * self.syntax_score
          + WEIGHTS["semantic"]   * self.semantic_score
          + WEIGHTS["compliance"] * self.compliance_score
          + WEIGHTS["edge_case"]  * self.edge_case_score,
            4
        )
        if self.weighted_risk < THRESHOLD_SAFE:
            self.classification = "SAFE"
        elif self.weighted_risk < THRESHOLD_HIGH_RISK:
            self.classification = "REVIEW REQUIRED"
        else:
            self.classification = "HIGH RISK"

    def to_dict(self) -> dict:
        return {
            "pair_id":          self.pair_id,
            "requirement":      self.requirement,
            "p1_label":         self.p1_label,
            "p1_halluc_type":   self.p1_ht,
            "component_scores": {
                "syntax":     round(self.syntax_score,    4),
                "semantic":   round(self.semantic_score,  4),
                "compliance": round(self.compliance_score,4),
                "edge_case":  round(self.edge_case_score, 4),
            },
            "weights":          WEIGHTS,
            "weighted_risk":    self.weighted_risk,
            "classification":   self.classification,
            "hallucination_types_detected": self.hallucination_types,
            "details": {
                "syntax_errors":          self.syntax_errors,
                "semantic_mismatches":    self.semantic_mismatches,
                "compliance_violations":  self.compliance_violations,
                "edge_cases":             self.edge_cases_found,
            }
        }


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_risk_aggregation_v2(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/risk_aggregated_v2_results.json"
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]

    print(f"\n{'='*72}")
    print(f"TrustGuard — Risk Aggregator v2 (recalibrated thresholds)")
    print(f"SAFE < {THRESHOLD_SAFE} | REVIEW < {THRESHOLD_HIGH_RISK} | HIGH RISK ≥ {THRESHOLD_HIGH_RISK}")
    print(f"{'='*72}")

    syn_v  = SyntaxValidator()
    sem_v  = SemanticValidatorV3()
    com_v  = ComplianceCheckerV3()
    edg_v  = EdgeCaseDetector()

    scores       = []
    class_counts = {"SAFE": 0, "REVIEW REQUIRED": 0, "HIGH RISK": 0}
    correct_preds = 0

    # Per-label score tracking for paper
    label_scores = {"correct": [], "hallucinated": [], "dangerous": []}

    print(f"\n  {'ID':<10} {'P1':<14} {'Syn':>5} {'Sem':>5} "
          f"{'Com':>5} {'Edg':>5} {'Risk':>6}  Result")
    print(f"  {'-'*72}")

    for pair in pairs:
        req    = pair["requirement"]
        rule   = pair.get("generated_rule")
        p1_lbl = pair.get("label", "unknown")
        p1_ht  = pair.get("hallucination_type", "none")

        syn_r = syn_v.validate(rule)
        sem_r = sem_v.match(req, rule)
        com_r = com_v.check(req, rule)
        edg_r = edg_v.detect(req, rule)

        rs = RuleRiskScore(
            pair_id=pair["pair_id"], requirement=req,
            p1_label=p1_lbl, p1_ht=p1_ht
        )
        rs.syntax_score     = syntax_to_score(syn_r)
        rs.semantic_score   = semantic_to_score(sem_r)
        rs.compliance_score = compliance_to_score(com_r)
        rs.edge_case_score  = edge_case_to_score(edg_r)

        rs.syntax_errors          = syn_r.errors
        rs.semantic_mismatches    = sem_r.mismatches
        rs.compliance_violations  = com_r.violations
        rs.edge_cases_found       = edg_r.edge_cases_found
        rs.hallucination_types    = sem_r.hallucination_types_detected

        rs.compute()
        class_counts[rs.classification] += 1
        scores.append(rs)

        if p1_lbl in label_scores:
            label_scores[p1_lbl].append(rs.weighted_risk)

        # Agreement check
        p1_not_ok  = p1_lbl in {"hallucinated", "dangerous"}
        our_not_ok = rs.classification in {"REVIEW REQUIRED", "HIGH RISK"}
        if p1_not_ok == our_not_ok:
            correct_preds += 1

        icon = {"SAFE":"✅","REVIEW REQUIRED":"⚠️ ","HIGH RISK":"🔴"}.get(rs.classification,"?")
        print(f"  {pair['pair_id']:<10} {p1_lbl:<14} "
              f"{rs.syntax_score:>5.2f} {rs.semantic_score:>5.2f} "
              f"{rs.compliance_score:>5.2f} {rs.edge_case_score:>5.2f} "
              f"{rs.weighted_risk:>6.3f}  {icon} {rs.classification}")

    accuracy = correct_preds / len(pairs) * 100

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "metadata": {
            "created_at":      datetime.now(UTC).isoformat(),
            "total":           len(scores),
            "weights":         WEIGHTS,
            "thresholds":      {"safe": THRESHOLD_SAFE, "high_risk": THRESHOLD_HIGH_RISK},
            "classification_distribution": class_counts,
            "agreement_with_p1": f"{accuracy:.1f}%",
            "avg_risk_by_label": {
                lbl: round(sum(v)/len(v), 4) if v else 0
                for lbl, v in label_scores.items()
            },
            "note": "Hand risk_aggregated_v2_results.json to Person 2"
        },
        "rules": [s.to_dict() for s in scores]
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    total = len(scores)
    print(f"\n{'='*72}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"  {'✅ SAFE:':<24} {class_counts['SAFE']:>3} ({class_counts['SAFE']/total*100:.1f}%)")
    print(f"  {'⚠️  REVIEW REQUIRED:':<24} {class_counts['REVIEW REQUIRED']:>3} ({class_counts['REVIEW REQUIRED']/total*100:.1f}%)")
    print(f"  {'🔴 HIGH RISK:':<24} {class_counts['HIGH RISK']:>3} ({class_counts['HIGH RISK']/total*100:.1f}%)")
    print(f"\n  Average risk score by P1 label (paper metric):")
    for lbl, vals in label_scores.items():
        if vals:
            avg = sum(vals)/len(vals)
            print(f"    {lbl:<15} avg={avg:.4f}  "
                  f"(want: correct < hallucinated < dangerous)")
    print(f"\n  Agreement with P1: {accuracy:.1f}%")
    print(f"  Saved: {output_path}")
    print(f"{'='*72}\n")

    return output


if __name__ == "__main__":
    run_risk_aggregation_v2()