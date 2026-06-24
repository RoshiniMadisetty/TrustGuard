
import json
with open('week6_final_report.json') as f:
    d = json.load(f)
print('FINAL REPORT KEYS:', list(d.keys()) if isinstance(d, dict) else type(d))

with open('week6_xai_disagreement.json') as f:
    x = json.load(f)
print('XAI KEYS:', list(x.keys()) if isinstance(x, dict) else type(x))

with open('week6_ensemble_confidence.json') as f:
    e = json.load(f)
print('ENSEMBLE KEYS:', list(e.keys()) if isinstance(e, dict) else type(e))
