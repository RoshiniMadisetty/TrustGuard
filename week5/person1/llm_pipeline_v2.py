"""
TrustGuard - Week 5 | Person 1: LLM Pipeline v2
------------------------------------------------
Structured Output Enforcer + Chain-of-Thought Prompt Engineering
- Forces JSON-schema-compliant firewall policy output from LLM
- Adds CoT reasoning trace for XAI handoff (Person 2)
- Retry logic with fallback parsing
- Outputs: week5_llm_outputs.json (handoff to Person 3)
"""

import json
import re
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("week5_person1.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("TrustGuard.P1")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "llama3.1:8b"
MAX_RETRIES  = 3
RETRY_DELAY  = 2   # seconds
TEMPERATURE  = 0.2  # low = more deterministic, reduces hallucination
INPUT_FILE   = "week4_final_dataset.json"
OUTPUT_FILE  = "week5_llm_outputs.json"

# ── Firewall Policy JSON Schema (strict) ─────────────────────────────────────
POLICY_SCHEMA = {
    "policy_id":     "string  — unique slug e.g. FW-001",
    "description":   "string  — plain-English intent",
    "action":        "ALLOW | DENY | DROP",
    "protocol":      "TCP | UDP | ICMP | ANY",
    "src_ip":        "CIDR or 'ANY'",
    "dst_ip":        "CIDR or 'ANY'",
    "src_port":      "integer 1-65535 or 'ANY'",
    "dst_port":      "integer 1-65535 or 'ANY'",
    "direction":     "INBOUND | OUTBOUND | BOTH",
    "priority":      "integer 1-1000 (lower = higher priority)",
    "reasoning":     "string  — step-by-step justification (CoT trace)",
    "confidence":    "float 0.0-1.0 — model self-assessed confidence"
}

# ── System Prompt (v2 with CoT + schema enforcement) ─────────────────────────
SYSTEM_PROMPT = """You are a senior network security engineer generating enterprise firewall policies.

CRITICAL RULES:
1. Respond ONLY with a single valid JSON object — no markdown, no explanation outside JSON.
2. The JSON MUST strictly follow this schema:
{schema}

3. For the "reasoning" field, provide explicit chain-of-thought:
   Step 1: Identify the threat or access requirement.
   Step 2: Choose the minimal-privilege action.
   Step 3: Specify exact protocol/port constraints.
   Step 4: Justify src/dst scope (why not broader).
   Step 5: State any caveats or edge cases.

4. Assign "confidence" honestly:
   - 1.0 = completely certain, no ambiguity
   - 0.7-0.9 = minor ambiguity in scope or protocol
   - <0.7 = significant ambiguity — flag in reasoning

5. NEVER use '0.0.0.0/0' as dst_ip unless the policy explicitly requires internet egress.
6. NEVER set action=ALLOW with src_ip=ANY and dst_ip=ANY simultaneously.
7. Default to DENY for ambiguous requests.
""".format(schema=json.dumps(POLICY_SCHEMA, indent=2))


# ── Core LLM Call ─────────────────────────────────────────────────────────────
def call_ollama(prompt: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Call Ollama and return raw text response with retry logic."""
    payload = {
        "model":  MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser Request:\n{prompt}",
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 1024,
            "stop": ["```"]          # prevent markdown fence leakage
        }
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            log.debug(f"Raw LLM response (attempt {attempt}): {raw[:200]}")
            return raw
        except Exception as e:
            log.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
    log.error("All retries exhausted.")
    return None


# ── JSON Extraction & Repair ──────────────────────────────────────────────────
def extract_json(raw: str) -> Optional[dict]:
    """
    Robustly extract JSON from LLM output.
    Handles: bare JSON, JSON in ```json blocks, leading text before '{'.
    """
    # 1. Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Extract first {...} block
    match = re.search(r'\{[\s\S]+\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # 3. Strip markdown fences
    clean = re.sub(r'```(?:json)?', '', raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    log.error(f"JSON extraction failed for output: {raw[:300]}")
    return None


# ── Schema Validation ─────────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "policy_id", "description", "action", "protocol",
    "src_ip", "dst_ip", "src_port", "dst_port",
    "direction", "priority", "reasoning", "confidence"
}

VALID_ACTIONS    = {"ALLOW", "DENY", "DROP"}
VALID_PROTOCOLS  = {"TCP", "UDP", "ICMP", "ANY"}
VALID_DIRECTIONS = {"INBOUND", "OUTBOUND", "BOTH"}

def validate_schema(policy: dict) -> tuple[bool, list[str]]:
    """Validate extracted JSON against firewall policy schema. Returns (valid, errors)."""
    errors = []

    # Missing fields
    missing = REQUIRED_FIELDS - set(policy.keys())
    if missing:
        errors.append(f"Missing fields: {missing}")

    # Enum checks
    if policy.get("action") not in VALID_ACTIONS:
        errors.append(f"Invalid action: {policy.get('action')}")
    if policy.get("protocol") not in VALID_PROTOCOLS:
        errors.append(f"Invalid protocol: {policy.get('protocol')}")
    if policy.get("direction") not in VALID_DIRECTIONS:
        errors.append(f"Invalid direction: {policy.get('direction')}")

    # Confidence range
    conf = policy.get("confidence")
    if conf is not None:
        try:
            c = float(conf)
            if not (0.0 <= c <= 1.0):
                errors.append(f"Confidence {c} out of range [0,1]")
        except (TypeError, ValueError):
            errors.append(f"Confidence not numeric: {conf}")

    # Security rules
    if (policy.get("action") == "ALLOW"
            and policy.get("src_ip") == "ANY"
            and policy.get("dst_ip") == "ANY"):
        errors.append("SECURITY VIOLATION: ALLOW with src_ip=ANY and dst_ip=ANY")

    # Reasoning CoT completeness (must mention all 5 steps)
    reasoning = policy.get("reasoning", "")
    step_count = sum(1 for i in range(1, 6) if f"Step {i}" in reasoning)
    if step_count < 3:
        errors.append(f"Incomplete CoT reasoning — only {step_count}/5 steps found")

    return len(errors) == 0, errors


# ── Per-Record Processing ─────────────────────────────────────────────────────
def process_record(record: dict, idx: int) -> dict:
    """Process one dataset record through the LLM pipeline."""
    prompt     = record.get("requirement", "")
    record_id  = record.get("pair_id", f"rec_{idx:04d}")
    label      = record.get("label", "unknown")

    log.info(f"[{idx+1}] Processing record {record_id} | label={label}")

    raw_output = call_ollama(prompt)

    result = {
        "record_id":        record_id,
        "prompt":           prompt,
        "ground_truth_label": label,
        "raw_llm_output":   raw_output,
        "parsed_policy":    None,
        "schema_valid":     False,
        "schema_errors":    [],
        "generation_meta": {
            "model":       MODEL,
            "temperature": TEMPERATURE,
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()
        }
    }

    if raw_output is None:
        result["schema_errors"] = ["LLM call failed — no output"]
        return result

    parsed = extract_json(raw_output)
    if parsed is None:
        result["schema_errors"] = ["JSON extraction failed"]
        return result

    valid, errors = validate_schema(parsed)
    result["parsed_policy"]  = parsed
    result["schema_valid"]   = valid
    result["schema_errors"]  = errors

    if valid:
        log.info(f"  ✓ Valid policy generated | confidence={parsed.get('confidence')}")
    else:
        log.warning(f"  ✗ Schema violations: {errors}")

    return result


# ── Batch Runner ──────────────────────────────────────────────────────────────
def run_pipeline(input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE):
    """Main pipeline: load dataset → generate → validate → save."""
    log.info("=" * 60)
    log.info("TrustGuard Week 5 | Person 1 | LLM Pipeline v2")
    log.info("=" * 60)

    # Load input dataset
    p = Path(input_path)
    if not p.exists():
        log.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)

    with open(p, "r") as f:
        raw_dataset = json.load(f)

    dataset = raw_dataset["pairs"]

    log.info(f"Loaded {len(dataset)} records")

    log.info(f"Loaded {len(dataset)} records from {input_path}")

    results      = []
    valid_count  = 0
    failed_count = 0

    for idx, record in enumerate(dataset):
        result = process_record(record, idx)
        results.append(result)
        if result["schema_valid"]:
            valid_count += 1
        else:
            failed_count += 1

    # Summary stats
    total = len(results)
    summary = {
        "pipeline_run": {
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "model":         MODEL,
            "total_records": total,
            "valid_outputs": valid_count,
            "failed_outputs": failed_count,
            "schema_validity_rate": round(valid_count / total, 4) if total else 0,
        },
        "records": results
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info(f"Pipeline complete.")
    log.info(f"  Total    : {total}")
    log.info(f"  Valid    : {valid_count} ({100*valid_count/total:.1f}%)")
    log.info(f"  Failed   : {failed_count}")
    log.info(f"  Output   : {output_path}")
    log.info("=" * 60)

    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()