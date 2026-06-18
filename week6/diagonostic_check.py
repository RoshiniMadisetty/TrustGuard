import json

with open('week6_benchmark_report.json') as f:
    d = json.load(f)

bc = d['binary_classification']
br = d['benchmark_run']
print("=== BENCHMARK ===")
for k,v in bc.items():
    print(f"  {k}: {v}")
print(f"  labelled: {br.get('labelled_records')}")
print(f"  hallucinated: {br.get('hallucinated')}")
print(f"  clean: {br.get('clean')}")