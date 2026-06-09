"""
TrustGuard - Week 3 (Person 1)
Model Response Logger

Reads prompt test results and produces a clean human-readable log
showing what the LLM actually said for each test case.

Run: python response_logger.py
(Run after prompt_test_suite.py)
"""

import json
import os
from datetime import datetime, UTC


def generate_response_log(
    results_path: str = "../data/week3_prompt_test_results.json",
    log_path: str = "../outputs/week3_response_log.txt"
) -> None:

    with open(results_path) as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    lines = []

    header = f"""
================================================================================
  TrustGuard — Week 3 Prompt Test Response Log
  Generated: {datetime.now(UTC).isoformat()}
  Total cases: {data['metadata']['total_cases']}
  Injection blocked: {data['metadata']['injection_blocked']}
  Ambiguous flagged: {data['metadata']['ambiguous_flagged']}
  Passed to LLM: {data['metadata']['passed_to_llm']}
================================================================================
"""
    lines.append(header)

    # Group by category
    by_category = {}
    for tc in data["test_cases"]:
        cat = tc["category"]
        by_category.setdefault(cat, []).append(tc)

    for category, cases in sorted(by_category.items()):
        lines.append(f"\n{'─'*80}")
        lines.append(f"  CATEGORY: {category.upper()}")
        lines.append(f"{'─'*80}")

        for tc in cases:
            status_emoji = {
                "BLOCKED_INJECTION": "🚫",
                "FLAGGED_AMBIGUOUS": "⚠️ ",
                "PASSED_TO_LLM":     "✅",
                "LLM_NO_JSON":       "❌",
                "LLM_ERROR":         "❌",
            }.get(tc["final_status"], "➡️ ")

            lines.append(f"\n  {tc['test_id']} | Risk: {tc['expected_risk'].upper()} | {status_emoji} {tc['final_status']}")
            lines.append(f"  Requirement:      {tc['requirement']}")
            lines.append(f"  Expected:         {tc['expected_behavior']}")

            pf = tc.get("pre_filter", {})
            if pf.get("injection_detected"):
                lines.append(f"  ⚠ Injection:     Matched pattern '{pf['injection_pattern']}'")
            if pf.get("ambiguity_issues"):
                lines.append(f"  ⚠ Ambiguity:     {', '.join(pf['ambiguity_issues'])}")

            rule = tc.get("llm_rule")
            if rule:
                lines.append(f"  LLM Rule:         {rule.get('action','?').upper()} "
                              f"{rule.get('protocol','?').upper()} "
                              f"{rule.get('source','?')} → "
                              f"{rule.get('destination','?')}:{rule.get('destination_port','?')}")
            elif tc.get("llm_output") and tc["final_status"] not in ["BLOCKED_INJECTION", "FLAGGED_AMBIGUOUS"]:
                lines.append(f"  LLM Raw Output:   {tc['llm_output'][:200]}")

    # Summary stats
    status_counts = {}
    for tc in data["test_cases"]:
        s = tc["final_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    lines.append(f"\n\n{'='*80}")
    lines.append("  FINAL STATUS BREAKDOWN")
    lines.append(f"{'='*80}")
    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status:<25} {count:>3} cases")
    lines.append("")

    log_text = "\n".join(lines)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)

    print(log_text)
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    generate_response_log()