"""
TrustGuard - Week 2
Hallucination Category Report Generator

Reads the labeled dataset and produces:
  1. A summary statistics report (JSON + printed table)
  2. A CSV export for easy spreadsheet analysis
  3. A category breakdown with examples

Run: python hallucination_report.py
"""

import json
import csv
import os
from collections import Counter


def generate_report(
    labeled_path: str = "../data/week2_labeled_dataset.json",
    report_path: str = "../outputs/hallucination_report.json",
    csv_path: str = "../outputs/benchmark_dataset.csv"
):
    """
    Generate the Week 2 hallucination category report.
    """
    with open(labeled_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    total = len(pairs)

    print(f"\n{'='*65}")
    print(f"  TrustGuard — Hallucination Category Report")
    print(f"  Total Pairs: {total}")
    print(f"{'='*65}\n")

    # ── Label distribution ────────────────────────────────────────────────────
    label_counts = Counter(p["label"] for p in pairs)
    print("LABEL DISTRIBUTION:")
    print(f"  ✅ Correct:      {label_counts.get('correct',0):3d} "
          f"({label_counts.get('correct',0)/total*100:.1f}%)")
    print(f"  ⚠️  Hallucinated: {label_counts.get('hallucinated',0):3d} "
          f"({label_counts.get('hallucinated',0)/total*100:.1f}%)")
    print(f"  🔴 Dangerous:    {label_counts.get('dangerous',0):3d} "
          f"({label_counts.get('dangerous',0)/total*100:.1f}%)")

    # ── Hallucination type breakdown ──────────────────────────────────────────
    halluc_counts = Counter(
        p["hallucination_type"] for p in pairs
        if p["hallucination_type"] != "none"
    )

    print(f"\nHALLUCINATION TYPE BREAKDOWN:")
    print(f"  {'Type':<30} {'Count':>5}  {'%':>6}")
    print(f"  {'-'*46}")
    for htype, count in halluc_counts.most_common():
        print(f"  {htype:<30} {count:>5}  {count/total*100:>5.1f}%")

    # ── Examples per category ─────────────────────────────────────────────────
    print(f"\nEXAMPLES BY HALLUCINATION TYPE:")
    seen_types = set()
    for pair in pairs:
        htype = pair["hallucination_type"]
        if htype != "none" and htype not in seen_types:
            seen_types.add(htype)
            rule = pair.get("generated_rule", {})
            print(f"\n  [{htype.upper()}]")
            print(f"  Requirement: {pair['requirement']}")
            print(f"  Generated:   {rule.get('action','?').upper()} "
                  f"{rule.get('protocol','?').upper()} "
                  f"{rule.get('source','?')} → "
                  f"{rule.get('destination','?')}:{rule.get('destination_port','?')}")
            print(f"  Reason:      {pair['label_reasons'][0] if pair.get('label_reasons') else 'N/A'}")

    # ── Save JSON report ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "summary": {
            "total_pairs": total,
            "label_distribution": dict(label_counts),
            "hallucination_type_distribution": dict(halluc_counts),
            "hallucination_rate": (total - label_counts.get("correct", 0)) / total,
            "dangerous_rate": label_counts.get("dangerous", 0) / total
        },
        "pairs_by_type": {}
    }
    for htype in halluc_counts:
        report["pairs_by_type"][htype] = [
            {
                "pair_id": p["pair_id"],
                "requirement": p["requirement"],
                "label": p["label"],
                "reasons": p.get("label_reasons", [])
            }
            for p in pairs if p["hallucination_type"] == htype
        ]

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\n✓ JSON report saved to: {report_path}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "requirement",
            "action", "protocol", "source", "destination",
            "source_port", "destination_port", "direction",
            "label", "hallucination_type", "label_confidence", "reasons"
        ])
        writer.writeheader()
        for p in pairs:
            rule = p.get("generated_rule") or {}
            writer.writerow({
                "pair_id": p["pair_id"],
                "requirement": p["requirement"],
                "action": rule.get("action", ""),
                "protocol": rule.get("protocol", ""),
                "source": rule.get("source", ""),
                "destination": rule.get("destination", ""),
                "source_port": rule.get("source_port", "any"),
                "destination_port": rule.get("destination_port", "any"),
                "direction": rule.get("direction", ""),
                "label": p.get("label", ""),
                "hallucination_type": p.get("hallucination_type", ""),
                "label_confidence": p.get("label_confidence", ""),
                "reasons": " | ".join(p.get("label_reasons", []))
            })
    print(f"✓ CSV benchmark saved to: {csv_path}")
    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    generate_report()