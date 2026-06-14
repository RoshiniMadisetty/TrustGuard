import json

with open('week6_edge_case_scores.json') as f:
    d = json.load(f)

labelled = [r for r in d['records'] if r.get('has_label')]
correct_fps = [r for r in labelled 
               if r.get('is_hallucinated')==0 
               and r.get('adjusted_risk_score',0) >= 0.10]

print(f"Correct records scoring >= 0.10 (false positives): {len(correct_fps)}")
for r in correct_fps:
    p = r.get('parsed_policy', {})
    print(f"  {r['record_id']} score={r['adjusted_risk_score']:.3f} "
          f"action={p.get('action')} src={p.get('src_ip')} "
          f"port={p.get('dst_port')} cat={r.get('detected_category')} "
          f"h_type={r.get('hallucination_type')}")