import json
with open('week6_decisions.json') as f:
    d = json.load(f)
decisions = d.get('decisions', [])
# show first 3 records and all their keys
for r in decisions[:3]:
    print(r.keys())
    print('  label-related:', {k:v for k,v in r.items() if any(x in k.lower() for x in ['hall','label','ground','correct'])})
    print()