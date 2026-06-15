import json

with open('week6_validation_results.json') as f:
    d = json.load(f)

for r in d['records']:
    if r['record_id'] in ('W2-041', 'W2-042'):
        p = r.get('parsed_policy', {})
        v = r.get('validation', {})
        print(f"{r['record_id']} h_type={r['hallucination_type']}")
        print(f"  src_ip raw='{p.get('src_ip')}' dst_ip raw='{p.get('dst_ip')}'")
        print(f"  detected_cat={r['detected_category']} risk={r['risk_score']}")
        print(f"  compliance violations={v.get('compliance',{}).get('violations')}")
        print()