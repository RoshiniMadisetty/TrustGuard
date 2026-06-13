# save as: convert_csv_to_dataset.py
# run from: week6 folder

import csv, json
from pathlib import Path

CSV_PATH = Path("week4_final_dataset.csv")  # adjust if needed
OUT_PATH = Path("week4_final_dataset.json")

# Find the CSV — check a few locations
for candidate in [
    Path("week4_final_dataset.csv"),
    Path("../datasets/benchmark_dataset.csv"),
    Path("../person1_llm_pipeline/datasets/benchmark_dataset.csv"),
]:
    if candidate.exists():
        CSV_PATH = candidate
        break

print(f"Reading: {CSV_PATH}")

pairs = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row.get("pair_id"):
            continue

        # Map CSV columns to the structure the adapter expects
        pairs.append({
            "pair_id":          row["pair_id"],
            "requirement":      row["requirement"],
            "label":            row["label"],
            "hallucination_type": row.get("hallucination_type", "none"),
            "label_confidence": float(row.get("label_confidence") or 0.8),
            "raw_llm_output":   "",
            "generated_rule": {
                "action":            row.get("action", "DENY"),
                "protocol":          row.get("protocol", "TCP"),
                "source":            row.get("source", "ANY"),
                "destination":       row.get("destination", "ANY"),
                "source_port":       row.get("source_port", "ANY"),
                "destination_port":  row.get("destination_port", "ANY"),
                "direction":         row.get("direction", "INBOUND"),
                "priority":          100,
                "description":       row.get("requirement", ""),
            },
            "generation_metadata": {
                "model":         "benchmark_csv",
                "parse_success": True,
                "timestamp":     "",
                "llm_backend":   "csv_import",
            },
            "label_reasons":    [f"Labelled from benchmark CSV as: {row.get('label','')}"],
            "security_impact":  row.get("reasons", ""),
        })

output = {"pairs": pairs}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"Written {len(pairs)} records -> {OUT_PATH}")

# Quick check on first record
print("\nFirst record generated_rule:")
import pprint
pprint.pprint(pairs[0]["generated_rule"])