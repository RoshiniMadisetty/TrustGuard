"""
TrustGuard - Week 1
LLM Pipeline: Natural Language → Firewall Rule JSON

Usage:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in .env or environment variables.
    Run: python llm_pipeline.py
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional

# ── Uncomment the client you want to use ──────────────────────────────────────
# Option A: OpenAI GPT-4
# pip install openai
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Option B: Open-source via Ollama (local, free)
# Install Ollama: https://ollama.ai → run: ollama pull mistral
# pip install ollama
# import ollama  # see call_llm_ollama() below

# ─────────────────────────────────────────────────────────────────────────────

from schema import SEED_REQUIREMENTS


# ──────────────────────────────────────────────
# SYSTEM PROMPT  (the core of your LLM engine)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a network security engineer. 
Your task is to convert natural language firewall requirements into structured firewall rules.

Always respond with ONLY a valid JSON object — no explanation, no markdown, no extra text.

The JSON must follow this exact schema:
{
  "action": "allow" | "deny" | "drop" | "reject",
  "protocol": "tcp" | "udp" | "icmp" | "any" | "http" | "https" | "ssh" | "rdp" | "dns" | "ftp" | "smtp",
  "source": "<IP, subnet CIDR, object name, or 'any'>",
  "destination": "<IP, subnet CIDR, object name, or 'any'>",
  "source_port": "<port number, range like 1024-65535, or 'any'>",
  "destination_port": "<port number, range, or 'any'>",
  "direction": "inbound" | "outbound" | "both",
  "description": "<one-line explanation of what this rule does>"
}

Rules:
- Be specific. Never use "any" unless the requirement truly means unrestricted.
- Use standard port numbers (e.g., 443 for HTTPS, 22 for SSH, 3389 for RDP).
- Map service names to correct protocols and ports.
- Apply the principle of least privilege.
"""

USER_PROMPT_TEMPLATE = """Convert this requirement into a firewall rule JSON:

Requirement: {requirement}
"""


# ──────────────────────────────────────────────
# LLM CALL FUNCTIONS
# ──────────────────────────────────────────────

def call_llm_openai(requirement: str, model: str = "gpt-4o") -> str:
    """
    Call OpenAI GPT-4 to generate a firewall rule.
    Returns raw LLM text response.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(requirement=requirement)}
        ],
        temperature=0.2,    # Low temperature = more deterministic, fewer hallucinations
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def call_llm_ollama(requirement: str, model: str = "mistral") -> str:
    """
    Call a local Ollama model (free, no API key needed).
    Install: https://ollama.ai
    Pull model: ollama pull mistral
    """
    import ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(requirement=requirement)}
        ],
        options={"temperature": 0.2}
    )
    return response['message']['content'].strip()


def call_llm_huggingface(requirement: str, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
    """
    Call a HuggingFace model locally.
    pip install transformers torch accelerate
    NOTE: Requires ~14GB RAM for 7B models.
    """
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    prompt = f"[INST] <<SYS>>{SYSTEM_PROMPT}<</SYS>>\n{USER_PROMPT_TEMPLATE.format(requirement=requirement)} [/INST]"
    result = pipe(prompt, max_new_tokens=300, temperature=0.2, do_sample=True)
    return result[0]['generated_text'].split("[/INST]")[-1].strip()


# ──────────────────────────────────────────────
# JSON PARSER
# ──────────────────────────────────────────────

def parse_llm_output(raw_output: str) -> Optional[dict]:
    """
    Parse raw LLM text into a Python dict.
    Handles cases where LLM adds markdown fences or extra text.
    """
    # Strip markdown code fences if present
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` fences)
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    # Find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse error: {e}")
        return None


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run_week1_pipeline(
    requirements: list[str],
    llm_backend: str = "openai",  # "openai" | "ollama" | "huggingface"
    model: str = "gpt-4o",
    output_path: str = "../data/week1_seed_dataset.json",
    delay_seconds: float = 1.0     # Rate limiting for API calls
) -> list[dict]:
    """
    Main pipeline: runs all requirements through LLM and saves output.
    
    Args:
        requirements:   List of natural language firewall requirements
        llm_backend:    Which LLM to use
        model:          Model name/version
        output_path:    Where to save the dataset JSON
        delay_seconds:  Sleep between API calls to avoid rate limits

    Returns:
        List of requirement-rule pair dicts
    """
    dataset = []
    total = len(requirements)

    print(f"\n{'='*60}")
    print(f"TrustGuard Week 1 - LLM Pipeline")
    print(f"Backend: {llm_backend} | Model: {model}")
    print(f"Processing {total} requirements...")
    print(f"{'='*60}\n")

    for i, requirement in enumerate(requirements, 1):
        pair_id = f"W1-{i:03d}"
        print(f"[{i}/{total}] {pair_id}: {requirement[:60]}...")

        # ── Step 1: Call LLM ──────────────────────────────────
        raw_output = None
        try:
            if llm_backend == "openai":
                raw_output = call_llm_openai(requirement, model)
            elif llm_backend == "ollama":
                raw_output = call_llm_ollama(requirement, model)
            elif llm_backend == "huggingface":
                raw_output = call_llm_huggingface(requirement, model)
            else:
                raise ValueError(f"Unknown LLM backend: {llm_backend}")
        except Exception as e:
            print(f"  ✗ LLM call failed: {e}")
            raw_output = None

        # ── Step 2: Parse JSON ────────────────────────────────
        generated_rule = parse_llm_output(raw_output) if raw_output else None

        if generated_rule:
            print(f"  ✓ Rule: {generated_rule.get('action','?').upper()} "
                  f"{generated_rule.get('protocol','?').upper()} "
                  f"{generated_rule.get('source','?')} → "
                  f"{generated_rule.get('destination','?')}:"
                  f"{generated_rule.get('destination_port','?')}")
        else:
            print(f"  ✗ Failed to parse rule")

        # ── Step 3: Build pair record ─────────────────────────
        pair = {
            "pair_id": pair_id,
            "requirement": requirement,
            "generated_rule": generated_rule,
            "raw_llm_output": raw_output,
            "generation_metadata": {
                "llm_backend": llm_backend,
                "model": model,
                "timestamp": datetime.utcnow().isoformat(),
                "parse_success": generated_rule is not None
            }
        }
        dataset.append(pair)

        # Rate limiting
        if i < total:
            time.sleep(delay_seconds)

    # ── Step 4: Save dataset ──────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "total_pairs": len(dataset),
                "successful_parses": sum(1 for p in dataset if p["generated_rule"]),
                "llm_backend": llm_backend,
                "model": model
            },
            "pairs": dataset
        }, f, indent=2)

    success_count = sum(1 for p in dataset if p["generated_rule"])
    print(f"\n{'='*60}")
    print(f"✓ Done! {success_count}/{total} rules successfully generated.")
    print(f"✓ Dataset saved to: {output_path}")
    print(f"{'='*60}\n")

    return dataset


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── CHOOSE YOUR BACKEND ──────────────────────────────────────
    # For OpenAI: set OPENAI_API_KEY in environment
    #   export OPENAI_API_KEY="sk-..."
    # For Ollama (local, free):
    #   Install from https://ollama.ai, then: ollama pull mistral

    run_week1_pipeline(
        requirements=SEED_REQUIREMENTS,
        llm_backend="ollama",       # Change to "openai" if you have an API key
        model="mistral",            # Change to "gpt-4o" for OpenAI
        output_path="../data/week1_seed_dataset.json",
        delay_seconds=0.5
    )