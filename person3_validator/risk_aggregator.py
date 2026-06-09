"""
TrustGuard - Risk Aggregator
Person 3 (feeds into Person 2's risk engine)

Combines outputs from all four validators into a single risk score per rule.

Formula:
  Risk = 0.20 * syntax_score
       + 0.30 * semantic_score
       + 0.25 * compliance_score
       + 0.25 * edge_case_score

Classification:
  0.00 – 0.35  →  SAFE
  0.35 – 0.65  →  REVIEW REQUIRED
  0.65 – 1.00  →  HIGH RISK

This module:
  - Runs all four validators on the dataset
  - Combines scores
  - Outputs risk_aggregated_results.json (input for Person 2's risk engine)

Run: python risk_aggregator.py
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC

# Add person3 dir to path so we can import validators directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from syntax_validator       import SyntaxValidator
from semantic_validator_v3  import SemanticValidatorV3
from compliance_checker_v3 import ComplianceCheckerV3
from edge_case_handler      import EdgeCaseDetector


# ──────────────────────────────────────────────
# WEIGHTS (tunable — will be calibrated in Week 6)
# ──────────────────────────────────────────────

WEIGHTS = {
    "syntax":     0.20,
    "semantic":   0.30,
    "compliance": 0.25,
    "edge_case":  0.25,
}

# Classification thresholds
THRESHOLD_SAFE        = 0.35
THRESHOLD_HIGH_RISK   = 0.65


# ──────────────────────────────────────────────
# RISK SCORE DATACLASS
# ──────────────────────────────────────────────

@dataclass
class RuleRiskScore:
    pair_id:    str
    requirement: str

    # Component scores (0.0 = no issue, 1.0 = worst)
    syntax_score:     float = 0.0
    semantic_score:   float = 0.0
    compliance_score: float = 0.0
    edge_case_score:  float = 0.0

    # Component details
    syntax_errors:      list = field(default_factory=list)
    semantic_mismatches: list = field(default_factory=list)
    compliance_violations: list = field(default_factory=list)
    edge_cases:         list = field(default_factory=list)

    # Final
    weighted_risk:    float = 0.0
    classification:   str   = "SAFE"
    p1_label:         str   = "unknown"
    p1_halluc_type:   str   = "none"

    def compute(self):
        self.weighted_risk = round(
            WEIGHTS["syntax"]     * self.syntax_score
            + WEIGHTS["semantic"] * self.semantic_score
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
            "pair_id":            self.pair_id,
            "requirement":        self.requirement,
            "p1_label":           self.p1_label,
            "p1_halluc_type":     self.p1_halluc_type,
            "component_scores": {
                "syntax":     round(self.syntax_score,     4),
                "semantic":   round(self.semantic_score,   4),
                "compliance": round(self.compliance_score, 4),
                "edge_case":  round(self.edge_case_score,  4),
            },
            "weights": WEIGHTS,
            "weighted_risk":    self.weighted_risk,
            "classification":   self.classification,
            "details": {
                "syntax_errors":          self.syntax_errors,
                "semantic_mismatches":    self.semantic_mismatches,
                "compliance_violations":  self.compliance_violations,
                "edge_cases":             self.edge_cases,
            }
        }


# ──────────────────────────────────────────────
# SCORING HELPERS
# ──────────────────────────────────────────────

def syntax_to_score(validation_result) -> float:
    """Convert SyntaxValidator result to 0–1 risk score."""
    if not validation_result.is_valid:
        # Scale by number of errors, cap at 1.0
        return min(1.0, len(validation_result.errors) * 0.3)
    elif validation_result.warnings:
        return min(0.3, len(validation_result.warnings) * 0.05)
    return 0.0


def semantic_to_score(intent_result) -> float:
    """Convert SemanticValidator result to 0–1 risk score."""
    # final_confidence is 0–1 where 1 = perfect match
    # We invert it: low confidence = high risk
    conf = intent_result.final_confidence
    base = round(1.0 - conf, 4)

    # Boost for critical mismatches
    critical_count = sum(
        1 for mm in intent_result.mismatches
        if mm.get("severity") == "critical"
    )
    return min(1.0, base + critical_count * 0.15)


def compliance_to_score(compliance_result) -> float:
    """Use risk_score_contribution directly (already 0–1)."""
    return compliance_result.risk_score_contribution


def edge_case_to_score(edge_result) -> float:
    """Convert EdgeCaseResult to 0–1 risk score."""
    if edge_result.clean:
        return 0.0
    score = 0.0
    for ec in edge_result.edge_cases_found:
        sev = ec.get("severity", "medium")
        score += {"critical": 0.4, "high": 0.25,
                  "medium": 0.15, "low": 0.05}.get(sev, 0.1)
    return min(1.0, score)


# ──────────────────────────────────────────────
# MAIN AGGREGATOR
# ──────────────────────────────────────────────

def run_risk_aggregation(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/risk_aggregated_results.json"
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]

    # Initialise all validators once
    print(f"\n{'='*68}")
    print(f"TrustGuard — Risk Aggregator")
    print(f"Weights: syntax={WEIGHTS['syntax']} semantic={WEIGHTS['semantic']} "
          f"compliance={WEIGHTS['compliance']} edge={WEIGHTS['edge_case']}")
    print(f"Dataset: {len(pairs)} pairs")
    print(f"{'='*68}")

    syntax_v    = SyntaxValidator()
    semantic_v  = SemanticValidatorV3()
    compliance_v = ComplianceCheckerV3()
    edge_v      = EdgeCaseDetector()

    scores      = []
    class_counts = {"SAFE": 0, "REVIEW REQUIRED": 0, "HIGH RISK": 0}

    # Agreement tracking vs P1
    correct_preds = 0

    print(f"\n  {'ID':<10} {'P1 Label':<14} {'Syn':>5} "
          f"{'Sem':>5} {'Com':>5} {'Edg':>5} {'Risk':>6}  Classification")
    print(f"  {'-'*75}")

    for pair in pairs:
        req    = pair["requirement"]
        rule   = pair.get("generated_rule")
        p1_lbl = pair.get("label", "unknown")
        p1_ht  = pair.get("hallucination_type", "none")

        # ── Run all four validators ───────────────────────────────────────────
        syn_r  = syntax_v.validate(rule)
        sem_r  = semantic_v.match(req, rule)
        com_r  = compliance_v.check(req, rule)
        edg_r  = edge_v.detect(req, rule)

        # ── Convert to scores ─────────────────────────────────────────────────
        rs = RuleRiskScore(
            pair_id     = pair["pair_id"],
            requirement = req,
            p1_label    = p1_lbl,
            p1_halluc_type = p1_ht,
        )
        rs.syntax_score     = syntax_to_score(syn_r)
        rs.semantic_score   = semantic_to_score(sem_r)
        rs.compliance_score = compliance_to_score(com_r)
        rs.edge_case_score  = edge_case_to_score(edg_r)

        # Details for downstream use by Person 2
        rs.syntax_errors         = syn_r.errors
        rs.semantic_mismatches   = sem_r.mismatches
        rs.compliance_violations = com_r.violations
        rs.edge_cases            = edg_r.edge_cases_found

        rs.compute()
        class_counts[rs.classification] += 1
        scores.append(rs)

        # Check if our classification agrees with P1
        p1_not_ok = p1_lbl in {"hallucinated","dangerous"}
        our_not_ok = rs.classification in {"REVIEW REQUIRED","HIGH RISK"}
        if p1_not_ok == our_not_ok:
            correct_preds += 1

        # Icons
        cls_icon = {"SAFE":"✅","REVIEW REQUIRED":"⚠️ ","HIGH RISK":"🔴"
                    }.get(rs.classification, "?")
        print(f"  {pair['pair_id']:<10} {p1_lbl:<14} "
              f"{rs.syntax_score:>5.2f} {rs.semantic_score:>5.2f} "
              f"{rs.compliance_score:>5.2f} {rs.edge_case_score:>5.2f} "
              f"{rs.weighted_risk:>6.3f}  "
              f"{cls_icon} {rs.classification}")

    # ── Save results ──────────────────────────────────────────────────────────
    accuracy = correct_preds / len(pairs) * 100
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "created_at":        datetime.now(UTC).isoformat(),
            "total":             len(scores),
            "weights":           WEIGHTS,
            "thresholds": {
                "safe":        THRESHOLD_SAFE,
                "high_risk":   THRESHOLD_HIGH_RISK
            },
            "classification_distribution": class_counts,
            "agreement_with_p1": f"{accuracy:.1f}%",
            "note": "This file is the input for Person 2's risk engine (person2_risk_engine/)"
        },
        "rules": [s.to_dict() for s in scores]
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    total = len(scores)
    print(f"\n{'='*68}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"  {'✅ SAFE:':<22} {class_counts['SAFE']:>3} "
          f"({class_counts['SAFE']/total*100:.1f}%)")
    print(f"  {'⚠️  REVIEW REQUIRED:':<22} {class_counts['REVIEW REQUIRED']:>3} "
          f"({class_counts['REVIEW REQUIRED']/total*100:.1f}%)")
    print(f"  {'🔴 HIGH RISK:':<22} {class_counts['HIGH RISK']:>3} "
          f"({class_counts['HIGH RISK']/total*100:.1f}%)")
    print(f"\n  Agreement with P1 labels: {accuracy:.1f}%")
    print(f"\n  Output saved to: {output_path}")
    print(f"  → Hand this file to Person 2 as input for SHAP/LIME risk engine")
    print(f"{'='*68}\n")

    return output


if __name__ == "__main__":
    run_risk_aggregation()