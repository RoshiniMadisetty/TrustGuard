with open('orchestrator.py', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines[379:425], start=380):
    print(f"{i}: {line}", end='')