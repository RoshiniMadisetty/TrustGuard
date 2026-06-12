"""
TrustGuard - LLM Rule Generator
--------------------------------
Reads your expanded benchmark CSV (225 rows),
calls Ollama to generate firewall rules for each requirement,
appends results into week4_final_dataset.json format.

CSV expected columns (flexible - auto-detected):
  requirement / prompt / rule_description  -> the natural language requirement
  label                                    -> correct / hallucinated / dangerous
  hallucination_type                       -> wrong_port / intent_flip / etc.
  label_confidence                         -> 0.0-1.0 (optional)

Output: week4_final_dataset.json (same format as original)
"""

import json, csv, time, hashlib, re, sys, os
import requests
from pathlib import Path
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "llama3.1:8b"
TEMPERATURE   = 0.2
MAX_RETRIES   = 3
RETRY_DELAY   = 2

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
CSV_FILE      = BASE_DIR / "benchmark_dataset.csv"      # your expanded CSV
OUTPUT_FILE   = BASE_DIR / "week4_final_dataset.json"   # existing + new records
CHECKPOINT    = BASE_DIR / "generation_checkpoint.json" # resume if interrupted

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a network security engineer generating firewall rules.
Given a natural language requirement, generate a JSON firewall rule.

RESPOND WITH ONLY a single valid JSON object, no explanation, no markdown:
{
  "action": "ALLOW" or "DENY" or "DROP",
  "protocol": "TCP" or "UDP" or "ICMP" or "ANY",
  "source": "<CIDR or ANY>",
  "destination": "<CIDR or ANY>",
  "source_port": "<integer or ANY>",
  "destination_port": "<integer or ANY>",
  "direction": "INBOUND" or "OUTBOUND" or "BOTH",
  "description": "<brief rule description>"
}

Rules:
- Use specific CIDRs where the requirement implies specificity
- Use correct ports: HTTP=80, HTTPS=443, SSH=22, DNS=53, RDP=3389, FTP=21
- Use correct protocols: DNS=UDP, NTP=UDP, HTTP/HTTPS/SSH=TCP
- NEVER use ALLOW with source=ANY and destination=ANY simultaneously
- Default to DENY for ambiguous requests
"""

# ── CSV column auto-detection ─────────────────────────────────────────────────
REQUIREMENT_COLS    = ["requirement","prompt","rule_description","description",
                       "policy","rule","scenario"]
LABEL_COLS          = ["label","class","category","type","ground_truth"]
HALL_TYPE_COLS      = ["hallucination_type","hall_type","error_type","category"]
CONFIDENCE_COLS     = ["label_confidence","confidence","score"]
PAIR_ID_COLS        = ["pair_id","id","record_id","row_id"]

def detect_col(headers: list, candidates: list) -> str:
    headers_lower = [h.lower().strip() for h in headers]
    for c in candidates:
        if c.lower() in headers_lower:
            return headers[headers_lower.index(c.lower())]
    return None

# ── LLM call ──────────────────────────────────────────────────────────────────
def call_ollama(requirement: str) -> str | None:
    payload = {
        "model":  MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nRequirement: {requirement}\n\nJSON rule:",
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": 512, "stop": ["```"]}
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None

def extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract first {...}
    match = re.search(r'\{[\s\S]+\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Strip markdown
    clean = re.sub(r'```(?:json)?', '', raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None

# ── Load existing dataset ─────────────────────────────────────────────────────
def load_existing() -> tuple[dict, set]:
    """Returns (dataset_dict, set_of_existing_pair_ids)"""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        existing_ids = {p.get("pair_id","") for p in data.get("pairs", [])}
        print(f"Loaded existing dataset: {len(data.get('pairs',[]))} records")
        return data, existing_ids
    else:
        # Create fresh dataset structure
        return {
            "metadata": {
                "project": "TrustGuard",
                "version": "2.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_pairs_analyzed": 0,
                "weeks_covered": ["Week 1","Week 2","Week 3","Week 4"],
                "ready_for_handoff": True
            },
            "dataset_summary": {},
            "hallucination_types": {},
            "pairs": []
        }, set()

# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return set(json.load(f).get("processed_ids", []))
    return set()

def save_checkpoint(processed_ids: set):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"processed_ids": list(processed_ids),
                   "updated_at": datetime.now(timezone.utc).isoformat()}, f)

# ── Process one CSV row ───────────────────────────────────────────────────────
def process_row(row: dict, col_map: dict, existing_ids: set,
                processed_ids: set, counter: dict) -> dict | None:
    requirement = row.get(col_map["requirement"], "").strip()
    if not requirement:
        return None

    label        = row.get(col_map.get("label",""), "correct").strip().lower()
    h_type       = row.get(col_map.get("hall_type",""), "none").strip().lower()
    confidence   = float(row.get(col_map.get("confidence",""), 0.8) or 0.8)

    # Generate stable pair_id from requirement hash
    pair_id = row.get(col_map.get("pair_id",""), "").strip()
    if not pair_id:
        hash_val = hashlib.md5(requirement.encode()).hexdigest()[:8].upper()
        week     = "W3" if counter["total"] >= 40 else "W2" if counter["total"] >= 20 else "W1"
        pair_id  = f"{week}-{hash_val}"

    # Skip if already processed or in existing dataset
    if pair_id in existing_ids or pair_id in processed_ids:
        print(f"  Skipping {pair_id} (already exists)")
        return None

    # Normalise label
    label_clean = label if label in ("correct","hallucinated","dangerous") else "correct"
    if label in ("wrong_port","wrong_protocol","intent_flip","scope_expansion",
                 "over_permissive","security_downgrade","missing_constraint"):
        label_clean = "hallucinated"
        if not h_type or h_type == "none":
            h_type = label

    print(f"  Generating [{pair_id}] label={label_clean} type={h_type}")
    print(f"    Req: {requirement[:70]}...")

    raw_output = call_ollama(requirement)
    parsed     = extract_json(raw_output) if raw_output else None
    parse_ok   = parsed is not None

    if not parse_ok:
        print(f"    [WARN] Parse failed for {pair_id}")
        parsed = {}

    # Normalise generated rule fields
    gen_rule = {
        "action":            str(parsed.get("action","DENY")).upper(),
        "protocol":          str(parsed.get("protocol","TCP")).upper(),
        "source":            str(parsed.get("source","ANY")),
        "destination":       str(parsed.get("destination","ANY")),
        "source_port":       parsed.get("source_port","ANY"),
        "destination_port":  parsed.get("destination_port","ANY"),
        "direction":         str(parsed.get("direction","INBOUND")).upper(),
        "description":       str(parsed.get("description", requirement))
    }

    record = {
        "pair_id":          pair_id,
        "requirement":      requirement,
        "generated_rule":   gen_rule,
        "raw_llm_output":   raw_output or "",
        "generation_metadata": {
            "llm_backend":  "ollama",
            "model":        MODEL,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "parse_success": parse_ok,
        },
        "label":            label_clean,
        "hallucination_type": h_type or "none",
        "label_confidence": confidence,
        "label_reasons":    [f"Labelled from benchmark CSV as: {label}"],
        "taxonomy_name":    h_type.replace("_"," ").title() if h_type != "none" else "Correct Rule",
        "security_impact":  _security_impact(label_clean, h_type),
        "compliance_violation": _compliance(h_type),
    }

    counter["total"]       += 1
    counter[label_clean]   = counter.get(label_clean, 0) + 1
    if parse_ok:
        counter["parse_ok"] = counter.get("parse_ok", 0) + 1

    print(f"    [OK] Generated rule: {gen_rule['action']} {gen_rule['protocol']} "
          f"{gen_rule['source']} -> {gen_rule['destination']}:{gen_rule['destination_port']}")
    return record

def _security_impact(label: str, h_type: str) -> str:
    impacts = {
        "over_permissive":    "Critical - creates open firewall holes",
        "intent_flip":        "Critical - blocks/allows opposite of intent",
        "wrong_port":         "High - rule targets wrong service port",
        "wrong_protocol":     "High - rule uses wrong protocol",
        "missing_constraint": "High - overly broad rule scope",
        "scope_expansion":    "High - internal service exposed externally",
        "security_downgrade": "Critical - secure intent mapped to insecure port",
        "none":               "None - correct rule",
    }
    return impacts.get(h_type, "Medium - potential policy violation")

def _compliance(h_type: str) -> list:
    violations = {
        "over_permissive":    ["Least Privilege", "Zero Trust"],
        "intent_flip":        ["Policy Integrity"],
        "wrong_port":         ["Service-Specific Access Control"],
        "wrong_protocol":     ["Protocol Compliance"],
        "missing_constraint": ["Least Privilege"],
        "scope_expansion":    ["Zero Trust", "Network Segmentation"],
        "security_downgrade": ["Encryption Requirements", "PCI-DSS"],
    }
    return violations.get(h_type, [])

# ── Update dataset summary ────────────────────────────────────────────────────
def update_summary(dataset: dict) -> dict:
    pairs  = dataset["pairs"]
    total  = len(pairs)
    labels = {}
    htypes = {}
    for p in pairs:
        l = p.get("label","unknown")
        h = p.get("hallucination_type","none")
        labels[l] = labels.get(l,0) + 1
        if h != "none":
            htypes[h] = htypes.get(h,0) + 1

    correct    = labels.get("correct",0)
    hallucin   = labels.get("hallucinated",0)
    dangerous  = labels.get("dangerous",0)

    dataset["metadata"]["total_pairs_analyzed"] = total
    dataset["metadata"]["updated_at"]           = datetime.now(timezone.utc).isoformat()
    dataset["dataset_summary"] = {
        "total_pairs":      total,
        "label_distribution": {
            "correct":          correct,
            "hallucinated":     hallucin,
            "dangerous":        dangerous,
            "correct_pct":      round(100*correct/total,1)    if total else 0,
            "hallucinated_pct": round(100*hallucin/total,1)   if total else 0,
            "dangerous_pct":    round(100*dangerous/total,1)  if total else 0,
        },
        "hallucination_rate": round(100*(hallucin+dangerous)/total,1) if total else 0,
        "hallucination_type_distribution": htypes,
        "parse_success_rate": round(
            100 * sum(1 for p in pairs
                      if p.get("generation_metadata",{}).get("parse_success",False))
            / total, 1) if total else 0,
    }
    return dataset

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TrustGuard - LLM Rule Generator")
    print("=" * 60)

    # Verify Ollama is running
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models",[])]
        print(f"Ollama running. Available models: {models}")
        if not any(MODEL.split(":")[0] in m for m in models):
            print(f"WARNING: {MODEL} not found. Run: ollama pull {MODEL}")
    except Exception as e:
        print(f"ERROR: Ollama not reachable: {e}")
        print("Start Ollama with: ollama serve")
        sys.exit(1)

    # Load CSV
    if not CSV_FILE.exists():
        print(f"ERROR: CSV not found at {CSV_FILE}")
        print(f"Rename your expanded CSV to: {CSV_FILE.name}")
        print(f"And place it at: {BASE_DIR}")
        sys.exit(1)

    with open(CSV_FILE, "r", encoding="utf-8-sig") as f:
        reader    = csv.DictReader(f)
        headers   = reader.fieldnames
        rows      = list(reader)

    print(f"CSV loaded: {len(rows)} rows | Columns: {headers}")

    # Auto-detect columns
    col_map = {
        "requirement": detect_col(headers, REQUIREMENT_COLS),
        "label":       detect_col(headers, LABEL_COLS),
        "hall_type":   detect_col(headers, HALL_TYPE_COLS),
        "confidence":  detect_col(headers, CONFIDENCE_COLS),
        "pair_id":     detect_col(headers, PAIR_ID_COLS),
    }
    print(f"Column mapping: {col_map}")

    if not col_map["requirement"]:
        print("ERROR: Could not find requirement/prompt column.")
        print(f"Available columns: {headers}")
        sys.exit(1)

    # Load existing data + checkpoint
    dataset, existing_ids  = load_existing()
    processed_ids          = load_checkpoint()
    already_done           = len(existing_ids) + len(processed_ids)
    new_rows               = [r for r in rows
                               if r.get(col_map["requirement"],"").strip()]
    print(f"Rows to process: {len(new_rows)} | Already done: {already_done}")

    counter = {"total": len(dataset.get("pairs",[])),
               "correct": 0, "hallucinated": 0, "dangerous": 0}
    new_records = []

    for i, row in enumerate(new_rows):
        print(f"\n[{i+1}/{len(new_rows)}]", end=" ")
        record = process_row(row, col_map, existing_ids, processed_ids, counter)
        if record:
            new_records.append(record)
            processed_ids.add(record["pair_id"])
            # Save checkpoint every 10 records
            if len(new_records) % 10 == 0:
                save_checkpoint(processed_ids)
                # Incremental save
                dataset["pairs"].extend(new_records)
                new_records = []
                dataset = update_summary(dataset)
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2)
                print(f"\n  [CHECKPOINT] Saved {len(dataset['pairs'])} records")

    # Final save
    if new_records:
        dataset["pairs"].extend(new_records)
    dataset = update_summary(dataset)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    save_checkpoint(processed_ids)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total records  : {dataset['dataset_summary']['total_pairs']}")
    print(f"Correct        : {dataset['dataset_summary']['label_distribution']['correct']}")
    print(f"Hallucinated   : {dataset['dataset_summary']['label_distribution']['hallucinated']}")
    print(f"Dangerous      : {dataset['dataset_summary']['label_distribution']['dangerous']}")
    print(f"Hallucination% : {dataset['dataset_summary']['hallucination_rate']}%")
    print(f"Parse success  : {dataset['dataset_summary']['parse_success_rate']}%")
    print(f"Output         : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()