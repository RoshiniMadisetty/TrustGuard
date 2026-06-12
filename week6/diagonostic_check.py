
import json

with open('week6_xai_disagreement.json') as f:
    d = json.load(f)

print('Per-record count:', len(d['per_record_analysis']))
print('Summary keys:', d['summary'].keys())
