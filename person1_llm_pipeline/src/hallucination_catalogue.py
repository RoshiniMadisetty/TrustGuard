"""
TrustGuard - Week 4 (Person 1)
Hallucination Catalogue Finaliser

Reads all outputs from Weeks 1-3 and produces:
  1. Final hallucination catalogue (JSON) — complete taxonomy with examples
  2. Team sync report (TXT) — summary for Person 2 and Person 3
  3. Validated dataset (JSON) — cleaned final version to hand off

Run: python hallucination_catalogue.py
"""

import json
import os
from datetime import datetime, UTC
from collections import Counter


# ──────────────────────────────────────────────
# HALLUCINATION TAXONOMY
# Full definitions for the paper + team reference
# ──────────────────────────────────────────────

HALLUCINATION_TAXONOMY = {
    "over_permissive": {
        "name": "Over-Permissive Rule",
        "definition": "The generated rule allows unrestricted access using 'any' for source, destination, or port when the requirement clearly implied specific constraints.",
        "security_impact": "Critical — creates open firewall holes exploitable by any attacker",
        "example_requirement": "Allow access to the server.",
        "example_generated": "ALLOW ANY any → any:any",
        "detection_method": "Check if action=allow AND source=any AND destination=any AND port=any",
        "compliance_violation": ["Least Privilege Principle", "Zero Trust"]
    },
    "intent_flip": {
        "name": "Intent Flip",
        "definition": "The generated rule uses the opposite action from what the requirement specified. A requirement to block becomes allow, or vice versa.",
        "security_impact": "Critical — directly contradicts the security intent of the policy",
        "example_requirement": "Block HTTP traffic and allow only HTTPS.",
        "example_generated": "ALLOW TCP any → any:80,443",
        "detection_method": "Check block/deny keywords in requirement vs allow action in rule, and vice versa",
        "compliance_violation": ["Security Policy Enforcement"]
    },
    "wrong_port": {
        "name": "Wrong Port Assignment",
        "definition": "The generated rule assigns an incorrect port number for the service mentioned in the requirement.",
        "security_impact": "High — rule either blocks legitimate traffic or opens wrong port",
        "example_requirement": "Allow employees to access the company website over HTTPS.",
        "example_generated": "ALLOW TCP 192.168.1.0/24 → web_server:80",
        "detection_method": "Map service keywords to expected ports and compare with generated destination_port",
        "compliance_violation": ["Service-Specific Access Control"]
    },
    "wrong_protocol": {
        "name": "Wrong Protocol",
        "definition": "The generated rule uses an incorrect protocol for the service. Common case: using TCP instead of ICMP for ping/block-ping requirements.",
        "security_impact": "Medium — rule may not function as intended",
        "example_requirement": "Block all ICMP ping requests from external sources.",
        "example_generated": "DENY TCP any → any:any",
        "detection_method": "Check service keywords (ping, icmp, dns) against generated protocol field",
        "compliance_violation": ["Protocol-Specific Policy"]
    },
    "missing_constraint": {
        "name": "Missing Source/Destination Constraint",
        "definition": "The requirement implied a specific source or destination restriction, but the generated rule defaults to 'any', removing the intended constraint.",
        "security_impact": "High — expands scope beyond what was authorised",
        "example_requirement": "Block all inbound SSH connections from the internet.",
        "example_generated": "DENY TCP 0.0.0.0/0 → 192.168.1.100:22",
        "detection_method": "Check restriction keywords (only, from, specific VLAN/group) against source/destination fields",
        "compliance_violation": ["Least Privilege Principle"]
    },
    "scope_expansion": {
        "name": "Scope Expansion",
        "definition": "The requirement mentioned a specific target system, but the generated rule applies to a much broader scope — often 'any' destination.",
        "security_impact": "High — rule affects more systems than intended",
        "example_requirement": "Deny direct internet access from the database server.",
        "example_generated": "DENY ANY db_server → 0.0.0.0/0:443,80,22,...",
        "detection_method": "Check if requirement names a specific target but rule uses any/0.0.0.0/0 for destination",
        "compliance_violation": ["Least Privilege Principle", "Network Segmentation"]
    },
    "security_downgrade": {
        "name": "Security Downgrade",
        "definition": "The requirement asked for a stronger security posture (e.g. HTTPS only), but the generated rule weakens it by also allowing the insecure variant (e.g. HTTP).",
        "security_impact": "High — reintroduces vulnerability the requirement was trying to close",
        "example_requirement": "Block HTTP and allow only HTTPS.",
        "example_generated": "ALLOW TCP any → any:80,443",
        "detection_method": "Check for HTTPS-only requirement but HTTP port (80) present in generated rule",
        "compliance_violation": ["Encryption Requirements", "PCI-DSS"]
    }
}


# ──────────────────────────────────────────────
# PROMPT INJECTION TAXONOMY
# ──────────────────────────────────────────────

INJECTION_TAXONOMY = {
    "direct_override": {
        "name": "Direct Override Attempt",
        "patterns": ["ignore previous instructions", "system override", "you are now"],
        "description": "Attacker directly tells the LLM to ignore its system prompt"
    },
    "permission_escalation": {
        "name": "Permission Escalation",
        "patterns": ["allow all traffic", "allow everything", "disable firewall"],
        "description": "Attacker disguises a dangerous rule as a legitimate requirement"
    },
    "context_manipulation": {
        "name": "Context Manipulation",
        "patterns": ["testing purposes", "temporarily allow", "for debugging", "maintenance mode"],
        "description": "Attacker adds fake justification to bypass security checks"
    },
    "embedded_json": {
        "name": "Embedded JSON Injection",
        "patterns": ["{action:", "output:"],
        "description": "Attacker embeds raw JSON in the requirement to override schema"
    }
}


# ──────────────────────────────────────────────
# MAIN CATALOGUE BUILDER
# ──────────────────────────────────────────────

def build_catalogue(
    labeled_path: str = "../data/week2_labeled_dataset.json",
    prompt_test_path: str = "../data/week3_prompt_test_results.json",
    catalogue_path: str = "../outputs/hallucination_catalogue.json",
    sync_report_path: str = "../outputs/team_sync_report.txt",
    final_dataset_path: str = "../data/week4_final_dataset.json"
):
    # ── Load Week 2 labeled data ──────────────────────────────────────────────
    with open(labeled_path) as f:
        labeled_data = json.load(f)
    pairs = labeled_data["pairs"]

    # ── Load Week 3 prompt test results ──────────────────────────────────────
    with open(prompt_test_path) as f:
        prompt_data = json.load(f)
    test_cases = prompt_data["test_cases"]

    print(f"\n{'='*65}")
    print(f"  TrustGuard Week 4 — Hallucination Catalogue Finaliser")
    print(f"{'='*65}\n")

    # ── Build per-type examples from real dataset ────────────────────────────
    examples_by_type = {}
    for pair in pairs:
        htype = pair.get("hallucination_type", "none")
        if htype == "none":
            continue
        if htype not in examples_by_type:
            examples_by_type[htype] = []
        rule = pair.get("generated_rule") or {}
        examples_by_type[htype].append({
            "pair_id": pair["pair_id"],
            "requirement": pair["requirement"],
            "generated_rule": f"{rule.get('action','?').upper()} "
                              f"{rule.get('protocol','?').upper()} "
                              f"{rule.get('source','?')} → "
                              f"{rule.get('destination','?')}:{rule.get('destination_port','?')}",
            "label": pair["label"],
            "reasons": pair.get("label_reasons", [])
        })

    # ── Stats ─────────────────────────────────────────────────────────────────
    label_counts = Counter(p["label"] for p in pairs)
    type_counts = Counter(
        p["hallucination_type"] for p in pairs
        if p["hallucination_type"] != "none"
    )
    injection_counts = Counter(
        tc["pre_filter"].get("injection_pattern", "")
        for tc in test_cases
        if tc.get("pre_filter", {}).get("injection_detected")
    )

    total = len(pairs)

    # ── Build catalogue ───────────────────────────────────────────────────────
    catalogue = {
        "metadata": {
            "project": "TrustGuard",
            "version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "total_pairs_analyzed": total,
            "weeks_covered": ["Week 1", "Week 2", "Week 3"],
            "ready_for_handoff": True
        },
        "dataset_summary": {
            "total_pairs": total,
            "label_distribution": {
                "correct": label_counts.get("correct", 0),
                "hallucinated": label_counts.get("hallucinated", 0),
                "dangerous": label_counts.get("dangerous", 0),
                "correct_pct": round(label_counts.get("correct", 0) / total * 100, 1),
                "hallucinated_pct": round(label_counts.get("hallucinated", 0) / total * 100, 1),
                "dangerous_pct": round(label_counts.get("dangerous", 0) / total * 100, 1),
            },
            "hallucination_rate": round(
                (total - label_counts.get("correct", 0)) / total * 100, 1
            )
        },
        "hallucination_types": {},
        "injection_analysis": {
            "total_test_cases": len(test_cases),
            "injection_blocked": prompt_data["metadata"]["injection_blocked"],
            "ambiguous_flagged": prompt_data["metadata"]["ambiguous_flagged"],
            "passed_to_llm": prompt_data["metadata"]["passed_to_llm"],
            "taxonomy": INJECTION_TAXONOMY,
            "detected_patterns": dict(injection_counts)
        }
    }

    # Fill in hallucination type entries
    for htype, taxonomy_entry in HALLUCINATION_TAXONOMY.items():
        count = type_counts.get(htype, 0)
        catalogue["hallucination_types"][htype] = {
            **taxonomy_entry,
            "observed_count": count,
            "observed_pct": round(count / total * 100, 1),
            "real_examples": examples_by_type.get(htype, [])[:3]  # max 3 examples
        }

    os.makedirs(os.path.dirname(catalogue_path), exist_ok=True)

    with open(catalogue_path, "w") as f:
        json.dump(catalogue, f, indent=2)
    print(f"✓ Hallucination catalogue saved: {catalogue_path}")

    # ── Build final validated dataset ─────────────────────────────────────────
    # Add catalogue reference to each pair
    final_pairs = []
    for pair in pairs:
        htype = pair.get("hallucination_type", "none")
        taxonomy_ref = HALLUCINATION_TAXONOMY.get(htype, {})
        final_pairs.append({
            **pair,
            "taxonomy_name": taxonomy_ref.get("name", "No Hallucination"),
            "security_impact": taxonomy_ref.get("security_impact", "None"),
            "compliance_violation": taxonomy_ref.get("compliance_violation", []),
            "week4_validated": True
        })

    final_dataset = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "total_pairs": len(final_pairs),
            "version": "final_week4",
            "handoff_ready": True
        },
        "pairs": final_pairs
    }

    with open(final_dataset_path, "w") as f:
        json.dump(final_dataset, f, indent=2)
    print(f"✓ Final validated dataset saved: {final_dataset_path}")

    # ── Build team sync report ────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  TrustGuard — Week 4 Team Sync Report")
    report_lines.append(f"  Generated: {datetime.now(UTC).isoformat()}")
    report_lines.append("=" * 70)

    report_lines.append(f"""
DATASET SUMMARY
  Total pairs:       {total}
  Correct:           {label_counts.get('correct', 0)} ({round(label_counts.get('correct',0)/total*100,1)}%)
  Hallucinated:      {label_counts.get('hallucinated', 0)} ({round(label_counts.get('hallucinated',0)/total*100,1)}%)
  Dangerous:         {label_counts.get('dangerous', 0)} ({round(label_counts.get('dangerous',0)/total*100,1)}%)
  Overall hallucination rate: {round((total - label_counts.get('correct',0))/total*100,1)}%
""")

    report_lines.append("HALLUCINATION TYPE BREAKDOWN")
    report_lines.append(f"  {'Type':<30} {'Count':>5}  {'%':>6}  {'Impact'}")
    report_lines.append(f"  {'-'*60}")
    for htype, count in type_counts.most_common():
        impact = HALLUCINATION_TAXONOMY.get(htype, {}).get("security_impact", "Unknown")
        report_lines.append(
            f"  {htype:<30} {count:>5}  {count/total*100:>5.1f}%  {impact}"
        )

    report_lines.append(f"""
PROMPT INJECTION ANALYSIS
  Total adversarial test cases:  {len(test_cases)}
  Injection attempts blocked:    {prompt_data['metadata']['injection_blocked']}
  Ambiguous inputs flagged:      {prompt_data['metadata']['ambiguous_flagged']}
  Cases passed to LLM:           {prompt_data['metadata']['passed_to_llm']}
""")

    report_lines.append("FILES HANDED OFF TO TEAM")
    report_lines.append("  File                              Goes to      Purpose")
    report_lines.append(f"  {'-'*65}")
    handoff_files = [
        ("week4_final_dataset.json",     "P2 + P3",   "Full labeled dataset with taxonomy"),
        ("benchmark_dataset.csv",        "P2 + P3",   "CSV for SHAP/LIME + validator testing"),
        ("hallucination_catalogue.json", "P2 + P3",   "Taxonomy reference for scoring engine"),
        ("week3_prompt_test_results.json","P2 + P3",  "Adversarial test cases + pre-filter results"),
        ("hallucination_report.json",    "P2 + P3",   "Category stats for paper"),
    ]
    for fname, dest, purpose in handoff_files:
        report_lines.append(f"  {fname:<35} {dest:<12} {purpose}")

    report_lines.append(f"""
INSTRUCTIONS FOR PERSON 2 (SHAP/LIME + Risk Engine)
  - Load week4_final_dataset.json as your primary input
  - Each pair has: requirement, generated_rule, label, hallucination_type,
    taxonomy_name, security_impact, compliance_violation
  - Apply SHAP + LIME to generated_rule fields
  - Use hallucination_type as ground truth for your confidence scoring
  - Risk scoring engine should map:
      dangerous   → High Risk
      hallucinated → Review Required
      correct     → Safe
  - Use hallucination_catalogue.json for taxonomy definitions in paper

INSTRUCTIONS FOR PERSON 3 (Syntax + Semantic Validator)
  - Load week4_final_dataset.json as your test dataset
  - Syntax validator: check generated_rule fields against valid schema
  - Semantic validator: compare requirement vs generated_rule
    and check if your flags match the label column
  - Use hallucination_type column to verify your detector catches the right category
  - Edge cases to handle are in pairs labeled:
      scope_expansion, missing_constraint, security_downgrade, contradictory_rule
  - Benchmark your precision/recall against the label column
""")

    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    with open(sync_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"✓ Team sync report saved: {sync_report_path}")

    # Print summary
    print(report_text)

    return catalogue


if __name__ == "__main__":
    build_catalogue()