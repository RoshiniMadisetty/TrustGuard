"""
TrustGuard — Person 3, Week 5
Integrated Validation Pipeline

Runs all four Person 3 validators in sequence on a single rule
and produces one unified output per rule.

Pipeline order:
  Input: requirement (str) + generated_rule (dict)
       ↓
  1. SyntaxValidator          → syntax_result
       ↓
  2. SemanticValidatorV3      → semantic_result
       ↓
  3. ComplianceCheckerV2      → compliance_result
       ↓
  4. EdgeCaseDetector         → edge_case_result
       ↓
  5. RiskAggregatorV2         → weighted_risk, classification
       ↓
  Output: ValidationReport (JSON)

Also confirms outputs feed correctly into Person 2's risk engine
by writing risk_aggregated_v2_results.json.

Run: python integrated_pipeline.py
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from syntax_validator      import SyntaxValidator
from semantic_validator_v3 import SemanticValidatorV3
from compliance_checker_v3 import ComplianceCheckerV3
from edge_case_handler     import EdgeCaseDetector

# ── Score conversion (same as risk_aggregator_v2) ─────────────────────────────
WEIGHTS = {"syntax": 0.20, "semantic": 0.30, "compliance": 0.25, "edge_case": 0.25}
THRESHOLD_SAFE      = 0.12
THRESHOLD_HIGH_RISK = 0.25

def _syntax_score(vr) -> float:
    if not vr.is_valid:
        return 0.90 if any("DANGEROUS" in e for e in vr.errors) else min(1.0, len(vr.errors)*0.40)
    return min(0.25, len(vr.warnings)*0.08) if vr.warnings else 0.0

def _semantic_score(imr) -> float:
    base     = round(1.0 - imr.final_confidence, 4)
    critical = sum(1 for mm in imr.mismatches if mm.get("severity")=="critical")
    high     = sum(1 for mm in imr.mismatches if mm.get("severity")=="high")
    return min(1.0, base + critical*0.25 + high*0.12)

def _compliance_score(cr) -> float:
    return cr.risk_score_contribution

def _edge_score(er) -> float:
    if er.clean: return 0.0
    score = sum({"critical":0.40,"high":0.25,"medium":0.15,"low":0.05}.get(
                 ec.get("severity","medium"),0.1) for ec in er.edge_cases_found)
    return min(1.0, score)

def _classify(risk: float) -> str:
    if risk < THRESHOLD_SAFE:      return "SAFE"
    if risk < THRESHOLD_HIGH_RISK: return "REVIEW REQUIRED"
    return "HIGH RISK"


# ──────────────────────────────────────────────
# VALIDATION REPORT
# ──────────────────────────────────────────────

@dataclass
class ValidationReport:
    pair_id:      str
    requirement:  str
    generated_rule: dict | None

    # Layer results
    syntax_valid:           bool  = True
    syntax_errors:          list  = field(default_factory=list)
    syntax_warnings:        list  = field(default_factory=list)

    semantic_match:         bool  = True
    semantic_mismatches:    list  = field(default_factory=list)
    semantic_confidence:    float = 1.0
    embedding_similarity:   float = -1.0

    compliance_ok:          bool  = True
    compliance_violations:  list  = field(default_factory=list)
    compliance_warnings:    list  = field(default_factory=list)

    edge_cases_clean:       bool  = True
    edge_cases_found:       list  = field(default_factory=list)

    # Scores
    syntax_score:     float = 0.0
    semantic_score:   float = 0.0
    compliance_score: float = 0.0
    edge_case_score:  float = 0.0
    weighted_risk:    float = 0.0
    classification:   str   = "SAFE"

    # Detected hallucination types
    hallucination_types: list = field(default_factory=list)

    # Ground truth (from Person 1)
    p1_label:     str = "unknown"
    p1_halluc_type: str = "none"

    def to_dict(self) -> dict:
        return {
            "pair_id":            self.pair_id,
            "requirement":        self.requirement,
            "p1_label":           self.p1_label,
            "p1_halluc_type":     self.p1_halluc_type,
            "layers": {
                "syntax": {
                    "valid":    self.syntax_valid,
                    "errors":   self.syntax_errors,
                    "warnings": self.syntax_warnings,
                    "score":    round(self.syntax_score, 4),
                },
                "semantic": {
                    "match":              self.semantic_match,
                    "mismatches":         self.semantic_mismatches,
                    "final_confidence":   round(self.semantic_confidence, 4),
                    "embedding_sim":      round(self.embedding_similarity, 4),
                    "score":              round(self.semantic_score, 4),
                },
                "compliance": {
                    "compliant":  self.compliance_ok,
                    "violations": self.compliance_violations,
                    "warnings":   self.compliance_warnings,
                    "score":      round(self.compliance_score, 4),
                },
                "edge_cases": {
                    "clean":       self.edge_cases_clean,
                    "cases_found": self.edge_cases_found,
                    "score":       round(self.edge_case_score, 4),
                },
            },
            "component_scores": {
                "syntax":     round(self.syntax_score,    4),
                "semantic":   round(self.semantic_score,  4),
                "compliance": round(self.compliance_score,4),
                "edge_case":  round(self.edge_case_score, 4),
            },
            "weights":          WEIGHTS,
            "weighted_risk":    round(self.weighted_risk, 4),
            "classification":   self.classification,
            "hallucination_types_detected": self.hallucination_types,
        }


# ──────────────────────────────────────────────
# INTEGRATED PIPELINE
# ──────────────────────────────────────────────

class TrustGuardPipeline:
    """
    Single entry point for all Person 3 validation layers.
    Validates one requirement + rule at a time.
    """

    def __init__(self):
        print("  Initialising TrustGuard validation pipeline...")
        self.syntax_v     = SyntaxValidator()
        self.semantic_v   = SemanticValidatorV3()
        self.compliance_v = ComplianceCheckerV3()
        self.edge_v       = EdgeCaseDetector()
        print("  ✓ All four validators loaded\n")

    def validate(self, pair_id: str, requirement: str,
                 rule: dict | None,
                 p1_label: str = "unknown",
                 p1_halluc_type: str = "none") -> ValidationReport:

        report = ValidationReport(
            pair_id=pair_id,
            requirement=requirement,
            generated_rule=rule,
            p1_label=p1_label,
            p1_halluc_type=p1_halluc_type,
        )

        # ── Layer 1: Syntax ───────────────────────────────────────────────────
        syn_r = self.syntax_v.validate(rule)
        report.syntax_valid    = syn_r.is_valid
        report.syntax_errors   = syn_r.errors
        report.syntax_warnings = syn_r.warnings
        report.syntax_score    = _syntax_score(syn_r)

        # ── Layer 2: Semantic ─────────────────────────────────────────────────
        sem_r = self.semantic_v.match(requirement, rule)
        report.semantic_match        = sem_r.matches
        report.semantic_mismatches   = sem_r.mismatches
        report.semantic_confidence   = sem_r.final_confidence
        report.embedding_similarity  = sem_r.embedding_similarity
        report.semantic_score        = _semantic_score(sem_r)
        report.hallucination_types   = sem_r.hallucination_types_detected[:]

        # ── Layer 3: Compliance ───────────────────────────────────────────────
        com_r = self.compliance_v.check(requirement, rule)
        report.compliance_ok         = com_r.compliant
        report.compliance_violations = com_r.violations
        report.compliance_warnings   = com_r.warnings
        report.compliance_score      = _compliance_score(com_r)

        # ── Layer 4: Edge Cases ───────────────────────────────────────────────
        edg_r = self.edge_v.detect(requirement, rule)
        report.edge_cases_clean  = edg_r.clean
        report.edge_cases_found  = edg_r.edge_cases_found
        report.edge_case_score   = _edge_score(edg_r)

        # Add edge-case hallucination types
        for ec in edg_r.edge_cases_found:
            ht = ec.get("type","")
            if ht and ht not in report.hallucination_types:
                report.hallucination_types.append(ht)

        # ── Aggregate ─────────────────────────────────────────────────────────
        report.weighted_risk = round(
            WEIGHTS["syntax"]     * report.syntax_score
          + WEIGHTS["semantic"]   * report.semantic_score
          + WEIGHTS["compliance"] * report.compliance_score
          + WEIGHTS["edge_case"]  * report.edge_case_score,
            4
        )
        report.classification = _classify(report.weighted_risk)

        return report


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_integrated_pipeline(
    dataset_path: str = None,
    output_path:  str = None,
    p2_feed_path: str = None,
):
    if dataset_path is None:
        dataset_path = os.path.join(ROOT_DIR, "person1_llm_pipeline",
                                    "data", "week4_final_dataset.json")
    if output_path is None:
        output_path  = os.path.join(THIS_DIR, "outputs",
                                    "integrated_pipeline_results.json")
    if p2_feed_path is None:
        p2_feed_path = os.path.join(THIS_DIR, "outputs",
                                    "risk_aggregated_v2_results.json")

    with open(dataset_path) as f:
        data = json.load(f)
    pairs = data["pairs"]

    pipeline = TrustGuardPipeline()
    reports  = []
    class_counts = {"SAFE": 0, "REVIEW REQUIRED": 0, "HIGH RISK": 0}
    correct_preds = 0

    print(f"{'='*70}")
    print(f"  TrustGuard Integrated Pipeline — {len(pairs)} rules")
    print(f"{'='*70}\n")
    print(f"  {'ID':<10} {'P1':<14} {'Syn':>5} {'Sem':>5} "
          f"{'Com':>5} {'Edg':>5} {'Risk':>6}  Result")
    print(f"  {'-'*70}")

    for pair in pairs:
        req    = pair["requirement"]
        rule   = pair.get("generated_rule")
        p1_lbl = pair.get("label","unknown")
        p1_ht  = pair.get("hallucination_type","none")

        rep = pipeline.validate(
            pair["pair_id"], req, rule, p1_lbl, p1_ht
        )
        class_counts[rep.classification] += 1
        reports.append(rep)

        p1_not_ok  = p1_lbl in {"hallucinated","dangerous"}
        our_not_ok = rep.classification in {"REVIEW REQUIRED","HIGH RISK"}
        if p1_not_ok == our_not_ok:
            correct_preds += 1

        icon = {"SAFE":"✅","REVIEW REQUIRED":"⚠️ ","HIGH RISK":"🔴"
                }.get(rep.classification,"?")
        print(f"  {pair['pair_id']:<10} {p1_lbl:<14} "
              f"{rep.syntax_score:>5.2f} {rep.semantic_score:>5.2f} "
              f"{rep.compliance_score:>5.2f} {rep.edge_case_score:>5.2f} "
              f"{rep.weighted_risk:>6.3f}  {icon} {rep.classification}")

    accuracy = correct_preds / len(pairs) * 100

    # ── Save integrated results ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    integrated_out = {
        "metadata": {
            "created_at":   datetime.now(UTC).isoformat(),
            "total":        len(reports),
            "weights":      WEIGHTS,
            "thresholds":   {"safe": THRESHOLD_SAFE, "high_risk": THRESHOLD_HIGH_RISK},
            "classification_distribution": class_counts,
            "agreement_with_p1": f"{accuracy:.1f}%",
        },
        "reports": [r.to_dict() for r in reports]
    }
    with open(output_path, "w") as f:
        json.dump(integrated_out, f, indent=2)
    print(f"\n  ✓ Integrated results → {output_path}")

    # ── Write P2 feed file (risk_aggregated_v2_results.json) ─────────────────
    # This is what Person 2's risk_engine_v2.py reads
    p2_rules = []
    for rep in reports:
        p2_rules.append({
            "pair_id":                    rep.pair_id,
            "requirement":                rep.requirement,
            "p1_label":                   rep.p1_label,
            "p1_halluc_type":             rep.p1_halluc_type,
            "component_scores": {
                "syntax":     round(rep.syntax_score,    4),
                "semantic":   round(rep.semantic_score,  4),
                "compliance": round(rep.compliance_score,4),
                "edge_case":  round(rep.edge_case_score, 4),
            },
            "weights":                    WEIGHTS,
            "weighted_risk":              round(rep.weighted_risk, 4),
            "classification":             rep.classification,
            "hallucination_types_detected": rep.hallucination_types,
            "details": {
                "syntax_errors":          rep.syntax_errors,
                "semantic_mismatches":    rep.semantic_mismatches,
                "compliance_violations":  rep.compliance_violations,
                "edge_cases":             rep.edge_cases_found,
            }
        })

    p2_feed = {
        "metadata": {
            "created_at":   datetime.now(UTC).isoformat(),
            "total":        len(p2_rules),
            "source":       "person3_validator/integrated_pipeline.py",
            "classification_distribution": class_counts,
            "agreement_with_p1": f"{accuracy:.1f}%",
            "note":         "Feed this file to person2_risk_engine/risk_engine_v2.py"
        },
        "rules": p2_rules
    }

    os.makedirs(os.path.dirname(p2_feed_path), exist_ok=True)
    with open(p2_feed_path, "w") as f:
        json.dump(p2_feed, f, indent=2)
    print(f"  ✓ P2 feed file     → {p2_feed_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(reports)
    print(f"\n{'='*70}")
    print(f"  CLASSIFICATION SUMMARY")
    print(f"  {'✅ SAFE:':<24} {class_counts['SAFE']:>3} ({class_counts['SAFE']/total*100:.1f}%)")
    print(f"  {'⚠️  REVIEW REQUIRED:':<24} {class_counts['REVIEW REQUIRED']:>3} ({class_counts['REVIEW REQUIRED']/total*100:.1f}%)")
    print(f"  {'🔴 HIGH RISK:':<24} {class_counts['HIGH RISK']:>3} ({class_counts['HIGH RISK']/total*100:.1f}%)")
    print(f"\n  Agreement with P1 labels: {accuracy:.1f}%")
    print(f"\n  Person 2 can now run:")
    print(f"    cd ../person2_risk_engine && python risk_engine_v2.py")
    print(f"{'='*70}\n")

    return integrated_out


if __name__ == "__main__":
    run_integrated_pipeline()