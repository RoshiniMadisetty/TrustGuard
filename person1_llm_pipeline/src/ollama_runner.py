"""
TrustGuard - Ollama JSON Runner
Replaces mock_runner.py with real Ollama calls that return clean JSON.

Install: pip install ollama
Model:   ollama pull llama3.1:8b  (already done)

Run: python ollama_runner.py
"""

import json
import os
import time
from datetime import datetime, UTC

try:
    import ollama
except ImportError:
    print("Run: pip install ollama")
    exit(1)

# ── The key fix: extremely strict JSON-only system prompt ─────────────────────
SYSTEM_PROMPT = """You are a firewall rule generator. You output ONLY valid JSON. No explanations. No markdown. No code blocks. No extra text. Just a single raw JSON object.

Output exactly this structure:
{"action":"allow or deny or drop","protocol":"tcp or udp or icmp or any","source":"IP or subnet or object or any","destination":"IP or subnet or object or any","source_port":"port number or any","destination_port":"port number or any","direction":"inbound or outbound or both","description":"one line explanation"}

Rules you must follow:
- Use "deny" not "block"
- Use exact port numbers: HTTPS=443, HTTP=80, SSH=22, RDP=3389, DNS=53, SMTP=587, FTP=21, Telnet=23, SNMP=161, SMB=445, PostgreSQL=5432, MySQL=3306
- Use "icmp" protocol for ping/ICMP, not tcp
- Never use "any" for source if requirement mentions a specific group or location
- Apply least privilege — be specific
- Output raw JSON only, starting with { and ending with }"""

USER_PROMPT = 'Requirement: {requirement}\n\nJSON:'


def call_ollama(requirement: str, model: str = "llama3.1:8b", retries: int = 3) -> dict | None:
    """
    Call Ollama and parse JSON. Retries up to 3 times if parsing fails.
    """
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(requirement=requirement)}
                ],
                options={
                    "temperature": 0.1,      # Very low — forces deterministic JSON
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )

            raw = response['message']['content'].strip()

            # Strip any markdown fences if model still adds them
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Find JSON boundaries
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"    Attempt {attempt}: No JSON found, retrying...")
                continue

            parsed = json.loads(raw[start:end])

            # Validate required fields exist
            required = ["action", "protocol", "source", "destination", "destination_port"]
            if all(k in parsed for k in required):
                return parsed
            else:
                print(f"    Attempt {attempt}: Missing fields, retrying...")

        except json.JSONDecodeError as e:
            print(f"    Attempt {attempt}: JSON parse error — {e}, retrying...")
        except Exception as e:
            print(f"    Attempt {attempt}: Error — {e}, retrying...")

        time.sleep(0.5)

    return None


def run_ollama_pipeline(
    requirements: list,
    model: str = "llama3.1:8b",
    output_path: str = "../data/week1_seed_dataset.json"
) -> list:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = []
    total = len(requirements)

    print(f"\n{'='*60}")
    print(f"TrustGuard — Ollama Pipeline (model: {model})")
    print(f"Processing {total} requirements...")
    print(f"{'='*60}\n")

    for i, requirement in enumerate(requirements, 1):
        print(f"[{i:02d}/{total}] {requirement[:65]}...")
        rule = call_ollama(requirement, model)

        if rule:
            print(f"        ✓ {rule.get('action','?').upper()} "
                  f"{rule.get('protocol','?').upper()} "
                  f"{rule.get('source','?')} → "
                  f"{rule.get('destination','?')}:{rule.get('destination_port','?')}")
        else:
            print(f"        ✗ Failed after retries — storing null")

        dataset.append({
            "pair_id": f"W1-{i:03d}",
            "requirement": requirement,
            "generated_rule": rule,
            "raw_llm_output": json.dumps(rule) if rule else None,
            "generation_metadata": {
                "llm_backend": "ollama",
                "model": model,
                "timestamp": datetime.now(UTC).isoformat(),
                "parse_success": rule is not None
            }
        })

    success = sum(1 for p in dataset if p["generated_rule"])
    output = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "total_pairs": len(dataset),
            "successful_parses": success,
            "llm_backend": "ollama",
            "model": model
        },
        "pairs": dataset
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {success}/{total} rules generated successfully.")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")
    return dataset


# ── 20 seed requirements ──────────────────────────────────────────────────────
SEED_REQUIREMENTS = [
    "Allow employees to access the company website over HTTPS.",
    "Block all inbound SSH connections from the internet.",
    "Allow the database server to communicate with the application server on port 5432.",
    "Deny all outbound traffic to known malicious IP ranges.",
    "Allow DNS queries from internal network to the corporate DNS server.",
    "Allow VPN users to access internal HR systems.",
    "Block RDP access from any external IP address.",
    "Allow IT admins to SSH into servers only from the management VLAN.",
    "Allow remote workers to connect to the file server over port 445.",
    "Deny any traffic from guest WiFi to the internal corporate network.",
    "Allow web servers to send outbound HTTPS requests to payment APIs.",
    "Block HTTP traffic and allow only HTTPS for all web communication.",
    "Allow the CI/CD pipeline server to pull code from GitHub over HTTPS.",
    "Deny direct internet access from the database server.",
    "Allow monitoring server to collect SNMP data from all network devices.",
    "Block all ICMP ping requests from external sources.",
    "Allow security scanning tools to probe internal servers on all ports.",
    "Deny any traffic on port 23 (Telnet) across the entire network.",
    "Allow email server to send outbound SMTP traffic on port 587.",
    "Block all traffic between the development environment and the production network.",
]

if __name__ == "__main__":
    run_ollama_pipeline(
        requirements=SEED_REQUIREMENTS,
        model="llama3.1:8b",
        output_path="../data/week1_seed_dataset.json"
    )