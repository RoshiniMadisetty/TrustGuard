"""
TrustGuard - Master Runner
Runs Week 1 + Week 2 pipeline end-to-end.

Run: python run_all.py
"""

import sys
import os

# ── Week 1 ──────────────────────────────────────────────────────────────────
print("\n" + "█"*65)
print("  WEEK 1: LLM Pipeline + Seed Dataset Generation")
print("█"*65)

sys.path.insert(0, "week1")
os.chdir("week1")
exec(open("mock_runner.py").read())   # Swap with llm_pipeline.py when you have API keys
os.chdir("..")

# ── Week 2 ──────────────────────────────────────────────────────────────────
print("\n" + "█"*65)
print("  WEEK 2: Dataset Expansion + Hallucination Labeling + Report")
print("█"*65)

sys.path.insert(0, "week2")
os.chdir("week2")

exec(open("dataset_expander.py").read())

from labeler import run_week2_labeling
run_week2_labeling(
    input_path="../data/week2_expanded_dataset.json",
    output_path="../data/week2_labeled_dataset.json"
)

from hallucination_report import generate_report
generate_report(
    labeled_path="../data/week2_labeled_dataset.json",
    report_path="../outputs/hallucination_report.json",
    csv_path="../outputs/benchmark_dataset.csv"
)

os.chdir("..")

print("\n" + "✅"*32)
print("  ALL DONE. Check data/ and outputs/ directories.")
print("✅"*32 + "\n")