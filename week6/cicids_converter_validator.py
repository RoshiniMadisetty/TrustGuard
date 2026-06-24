"""
CICIDS → TrustGuard JSON Converter & Validator
===============================================
Converts CICIDS2017/2018 network flow CSV data into
TrustGuard-compatible firewall rule JSON, then validates
each record against the TrustGuard schema.

Pipeline:
    CICIDS CSV → Flow Analysis → NLP Rule Synthesis → JSON → Validation

Supports two modes:
  --mode convert   : Full conversion (requires Ollama running)
  --mode validate  : Validate an already-converted JSON file
  --mode template  : Just show conversion template without Ollama

Usage:
    python cicids_converter_validator.py --mode template
    python cicids_converter_validator.py --mode convert --input cicids.csv --output converted.json
    python cicids_converter_validator.py --mode validate --input converted.json

CICIDS expected columns (CICIDS2017 format):
    Flow ID, Source IP, Source Port, Destination IP, Destination Port,
    Protocol, Timestamp, Flow Duration, Total Fwd Packets, Total Bwd Packets,
    Label (BENIGN / DDoS / PortScan / Bot / etc.)
"""

import os
import sys
import json
import csv
import argparse
import hashlib
import random
import re
from pathlib import Path
from datetime import datetime


# ─── CICIDS PROTOCOL MAP ─────────────────────────────────────────────────────

PROTOCOL_MAP = {
    "6":   "TCP",
    "17":  "UDP",
    "1":   "ICMP",
    "0":   "ANY",
    "tcp": "TCP",
    "udp": "UDP",
    "icmp": "ICMP",
}

# CICIDS attack labels → TrustGuard hallucination categories
LABEL_TO_CATEGORY = {
    "BENIGN":                None,               # correct rule
    "DDoS":                  "scope_expansion",
    "PortScan":              "wrong_port",
    "Bot":                   "intent_flip",
    "Web Attack – Brute Force": "security_downgrade",
    "Web Attack – XSS":      "security_downgrade",
    "Web Attack – Sql Injection": "security_downgrade",
    "FTP-Patator":           "wrong_protocol",
    "SSH-Patator":           "wrong_protocol",
    "DoS slowloris":         "scope_expansion",
    "DoS Slowhttptest":      "scope_expansion",
    "DoS Hulk":              "scope_expansion",
    "DoS GoldenEye":         "scope_expansion",
    "Heartbleed":            "missing_constraint",
    "Infiltration":          "over_permissive",
}

# Well-known port → service name
PORT_SERVICES = {
    "20": "FTP-data", "21": "FTP", "22": "SSH", "23": "Telnet",
    "25": "SMTP", "53": "DNS", "67": "DHCP", "68": "DHCP",
    "80": "HTTP", "110": "POP3", "143": "IMAP", "443": "HTTPS",
    "445": "SMB", "3306": "MySQL", "3389": "RDP", "5432": "PostgreSQL",
    "6379": "Redis", "8080": "HTTP-alt", "8443": "HTTPS-alt",
    "27017": "MongoDB",
}

# ─── TRUSTGUARD SCHEMA ───────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "record_id", "rule_id", "action", "protocol", "src_ip", "src_port",
    "dst_ip", "dst_port", "direction", "intent_description",
    "generated_rule", "hallucination_label", "hallucination_category",
    "confidence", "reasoning"
]

OPTIONAL_FIELDS = [
    "priority", "chain_of_thought", "semantic_score",
    "compliance_tags", "original_flow_label", "source_dataset"
]

VALID_ACTIONS    = {"ALLOW", "DENY", "BLOCK", "DROP", "REJECT"}
VALID_PROTOCOLS  = {"TCP", "UDP", "ICMP", "ANY"}
VALID_DIRECTIONS = {"inbound", "outbound", "both", "forward"}
VALID_HALL_LABELS = {"correct", "hallucinated"}
VALID_CATEGORIES = {
    "wrong_port", "wrong_protocol", "intent_flip", "scope_expansion",
    "over_permissive", "security_downgrade", "missing_constraint", None
}


# ─── ROW PARSER ──────────────────────────────────────────────────────────────

def parse_cicids_row(row, idx):
    """Extract key fields from a CICIDS CSV row (handles both 2017 and 2018 column names)."""

    def get(keys, default="ANY"):
        for k in keys:
            val = row.get(k, row.get(k.strip(), "")).strip()
            if val:
                return val
        return default

    src_ip    = get([" Source IP", "Src IP", "source_ip", "src_ip"], "10.0.0.1")
    dst_ip    = get([" Destination IP", "Dst IP", "destination_ip", "dst_ip"], "10.0.0.2")
    src_port  = get([" Source Port", "Src Port", "source_port", "src_port"], "ANY")
    dst_port  = get([" Destination Port", "Dst Port", "destination_port", "dst_port"], "ANY")
    protocol  = get([" Protocol", "Protocol", "protocol"], "6")
    label     = get([" Label", "Label", "label", " label"], "BENIGN").strip()
    flow_dur  = get([" Flow Duration", "Flow Duration", "flow_duration"], "0")
    fwd_pkts  = get([" Total Fwd Packets", "Total Fwd Packets", "fwd_packets"], "0")
    bwd_pkts  = get([" Total Backward Packets", "Total Bwd Packets", "bwd_packets"], "0")

    # Normalize protocol
    proto_str = PROTOCOL_MAP.get(protocol.lower(), PROTOCOL_MAP.get(protocol, "TCP"))

    # Normalize label
    label_clean = label.strip().upper()
    is_benign = label_clean == "BENIGN"

    # Map to hallucination category
    hall_category = None
    for key, cat in LABEL_TO_CATEGORY.items():
        if key.upper() in label_clean:
            hall_category = cat
            break

    # Determine action: attacks that are blocking events → DENY; benign → ALLOW
    action = "ALLOW" if is_benign else "DENY"

    # Service hint
    service = PORT_SERVICES.get(dst_port, f"port-{dst_port}")

    # Intent description from flow context
    if is_benign:
        intent = f"Allow {proto_str} traffic from {src_ip}:{src_port} to {dst_ip}:{dst_port} ({service})"
    else:
        intent = f"Block {label} attack: {proto_str} from {src_ip}:{src_port} to {dst_ip}:{dst_port} ({service})"

    # Synthesize a firewall rule string
    rule = {
        "action":    action,
        "protocol":  proto_str,
        "src_ip":    src_ip,
        "src_port":  src_port,
        "dst_ip":    dst_ip,
        "dst_port":  dst_port,
        "direction": "inbound" if not is_benign else "both",
    }

    # Reasoning (compact CoT)
    if is_benign:
        reasoning = (
            f"Flow from {src_ip}:{src_port} to {dst_ip}:{dst_port} via {proto_str} "
            f"classified BENIGN ({fwd_pkts} fwd pkts, {bwd_pkts} bwd pkts). "
            f"Standard ALLOW rule generated."
        )
    else:
        reasoning = (
            f"Flow classified as {label}. {proto_str} traffic from {src_ip}:{src_port} "
            f"to {dst_ip}:{dst_port}. Flow duration={flow_dur}μs, "
            f"{fwd_pkts} fwd + {bwd_pkts} bwd packets. DENY rule generated to block attack vector."
        )

    # Confidence: benign flows are higher confidence; attack detections vary
    base_conf = 0.85 if is_benign else 0.72
    conf = round(base_conf + random.uniform(-0.05, 0.05), 4)

    record_id = f"CICIDS-{idx:05d}"
    rule_id   = f"R-{hashlib.md5(f'{src_ip}{dst_ip}{dst_port}{proto_str}'.encode()).hexdigest()[:8].upper()}"

    return {
        "record_id":             record_id,
        "rule_id":               rule_id,
        "action":                action,
        "protocol":              proto_str,
        "src_ip":                src_ip,
        "src_port":              src_port,
        "dst_ip":                dst_ip,
        "dst_port":              dst_port,
        "direction":             rule["direction"],
        "intent_description":    intent,
        "generated_rule":        json.dumps(rule),
        "hallucination_label":   "correct" if is_benign else "hallucinated",
        "hallucination_category": hall_category,
        "confidence":            conf,
        "reasoning":             reasoning,
        "priority":              1 if not is_benign else 5,
        "chain_of_thought":      f"Step1: Parse flow. Step2: Classify label={label}. Step3: Map to firewall action={action}.",
        "semantic_score":        round(0.80 + random.uniform(-0.1, 0.1), 4),
        "compliance_tags":       ["LEAST_PRIVILEGE"] if is_benign else ["ZERO_TRUST", "DENY_DEFAULT"],
        "original_flow_label":   label,
        "source_dataset":        "CICIDS2017",
    }


# ─── OLLAMA NLP SYNTHESIS (optional enrichment) ──────────────────────────────

OLLAMA_PROMPT_TEMPLATE = """You are a firewall policy expert. Convert this network flow into a precise firewall rule JSON.

Flow data:
- Source IP: {src_ip}:{src_port}
- Destination IP: {dst_ip}:{dst_port}  
- Protocol: {protocol}
- Traffic Label: {label}
- Flow Duration: {flow_dur} microseconds

Return ONLY valid JSON with these exact fields:
{{
  "action": "ALLOW or DENY",
  "protocol": "{protocol}",
  "src_ip": "{src_ip}",
  "src_port": "{src_port}",
  "dst_ip": "{dst_ip}",
  "dst_port": "{dst_port}",
  "direction": "inbound or outbound",
  "intent": "one sentence describing what this rule does",
  "reasoning": "two sentences explaining why this action was chosen"
}}"""


def enrich_with_ollama(record, model="llama3.1:8b"):
    """Optional: Use Ollama to generate richer intent/reasoning. Requires Ollama running."""
    try:
        import urllib.request
        rule = json.loads(record["generated_rule"])
        prompt = OLLAMA_PROMPT_TEMPLATE.format(
            src_ip=rule["src_ip"], src_port=rule["src_port"],
            dst_ip=rule["dst_ip"], dst_port=rule["dst_port"],
            protocol=rule["protocol"],
            label=record["original_flow_label"],
            flow_dur="unknown"
        )
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            text = result.get("response", "").strip()
            # Extract JSON from response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                enriched = json.loads(match.group())
                if enriched.get("intent"):
                    record["intent_description"] = enriched["intent"]
                if enriched.get("reasoning"):
                    record["reasoning"] = enriched["reasoning"]
    except Exception:
        pass  # fallback to template-generated values
    return record


# ─── VALIDATOR ───────────────────────────────────────────────────────────────

def validate_record(record, idx):
    errors = []
    warnings = []

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None or record[field] == "":
            errors.append(f"Missing required field: '{field}'")

    # Action
    action = str(record.get("action", "")).upper()
    if action and action not in VALID_ACTIONS:
        errors.append(f"Invalid action '{action}'. Must be one of {VALID_ACTIONS}")

    # Protocol
    proto = str(record.get("protocol", "")).upper()
    if proto and proto not in VALID_PROTOCOLS:
        warnings.append(f"Unusual protocol '{proto}'. Expected {VALID_PROTOCOLS}")

    # Direction
    direction = str(record.get("direction", "")).lower()
    if direction and direction not in VALID_DIRECTIONS:
        warnings.append(f"Unusual direction '{direction}'. Expected {VALID_DIRECTIONS}")

    # Hallucination label
    hall_label = str(record.get("hallucination_label", "")).lower()
    if hall_label and hall_label not in VALID_HALL_LABELS:
        errors.append(f"Invalid hallucination_label '{hall_label}'. Must be 'correct' or 'hallucinated'")

    # Hallucination category consistency
    hall_cat = record.get("hallucination_category")
    if hall_label == "hallucinated" and hall_cat is None:
        warnings.append("hallucination_label=hallucinated but hallucination_category is None")
    if hall_label == "correct" and hall_cat is not None:
        warnings.append(f"hallucination_label=correct but hallucination_category='{hall_cat}'")
    if hall_cat not in VALID_CATEGORIES:
        errors.append(f"Invalid hallucination_category '{hall_cat}'")

    # Confidence range
    conf = record.get("confidence")
    if conf is not None:
        try:
            conf_f = float(conf)
            if not (0.0 <= conf_f <= 1.0):
                errors.append(f"confidence={conf_f} out of range [0,1]")
        except (ValueError, TypeError):
            errors.append(f"confidence must be numeric, got '{conf}'")

    # generated_rule should be valid JSON
    gr = record.get("generated_rule")
    if gr:
        if isinstance(gr, str):
            try:
                parsed = json.loads(gr)
                for key in ["action", "protocol", "src_ip", "dst_ip"]:
                    if key not in parsed:
                        warnings.append(f"generated_rule missing field '{key}'")
            except json.JSONDecodeError:
                errors.append("generated_rule is not valid JSON")
        elif not isinstance(gr, dict):
            errors.append("generated_rule must be a JSON string or dict")

    # Intent description length
    intent = record.get("intent_description", "")
    if intent and len(intent) < 10:
        warnings.append("intent_description is very short (< 10 chars)")

    return errors, warnings


def validate_dataset(records):
    results = {
        "total": len(records),
        "valid": 0,
        "with_warnings": 0,
        "invalid": 0,
        "records": []
    }

    label_counts = {"correct": 0, "hallucinated": 0}
    category_counts = {}

    for idx, rec in enumerate(records):
        errors, warnings = validate_record(rec, idx)
        status = "VALID" if not errors else "INVALID"
        if not errors and warnings:
            status = "VALID_WITH_WARNINGS"
            results["with_warnings"] += 1
        elif not errors:
            results["valid"] += 1
        else:
            results["invalid"] += 1

        # Count labels
        lbl = str(rec.get("hallucination_label", "")).lower()
        if lbl in label_counts:
            label_counts[lbl] += 1

        cat = rec.get("hallucination_category")
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        if errors or warnings:
            results["records"].append({
                "record_id": rec.get("record_id", f"idx-{idx}"),
                "status": status,
                "errors": errors,
                "warnings": warnings
            })

    results["label_distribution"] = label_counts
    results["category_distribution"] = category_counts
    results["compatibility_score"] = round(
        (results["valid"] + results["with_warnings"]) / results["total"] * 100, 2
    ) if results["total"] else 0

    return results


# ─── MAIN CONVERTER ──────────────────────────────────────────────────────────

def convert_cicids(input_path, output_path, max_records=None, use_ollama=False,
                   ollama_model="llama3.1:8b", balance_classes=True):
    print(f"Reading CICIDS CSV: {input_path}")
    records = []
    skipped = 0
    idx = 1

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_records and idx > max_records:
                break
            try:
                rec = parse_cicids_row(row, idx)
                if use_ollama:
                    rec = enrich_with_ollama(rec, ollama_model)
                records.append(rec)
                idx += 1
            except Exception as e:
                skipped += 1
                continue

    print(f"  Parsed: {len(records)} records | Skipped: {skipped}")

    # Class balance report
    benign_n  = sum(1 for r in records if r["hallucination_label"] == "correct")
    attack_n  = sum(1 for r in records if r["hallucination_label"] == "hallucinated")
    print(f"  Class balance: BENIGN={benign_n} | ATTACK={attack_n}")

    if balance_classes and attack_n > 0 and benign_n > 0:
        ratio = benign_n / attack_n
        if ratio > 3:
            print(f"  ⚠️  Imbalanced (ratio={ratio:.1f}x). Subsampling benign to 3x attacks.")
            benign_records = [r for r in records if r["hallucination_label"] == "correct"]
            attack_records = [r for r in records if r["hallucination_label"] == "hallucinated"]
            random.shuffle(benign_records)
            benign_records = benign_records[:min(len(benign_records), attack_n * 3)]
            records = benign_records + attack_records
            random.shuffle(records)
            print(f"  Balanced: {len(records)} records")

    output = {
        "dataset_info": {
            "source": "CICIDS2017",
            "converted_at": datetime.now().isoformat(),
            "total_records": len(records),
            "labelled": len(records),
            "unlabelled": 0,
            "correct": sum(1 for r in records if r["hallucination_label"] == "correct"),
            "hallucinated": sum(1 for r in records if r["hallucination_label"] == "hallucinated"),
        },
        "records": records
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  ✅ Saved: {output_path}")
    return records


# ─── REPORT PRINTER ──────────────────────────────────────────────────────────

def print_validation_report(vr, output_path=None):
    print()
    print("=" * 60)
    print("CICIDS → TrustGuard Compatibility Report")
    print("=" * 60)
    print(f"Total records    : {vr['total']}")
    print(f"Valid            : {vr['valid']}")
    print(f"Valid w/ warnings: {vr['with_warnings']}")
    print(f"Invalid          : {vr['invalid']}")
    print(f"Compatibility    : {vr['compatibility_score']}%")
    print()
    print("Label distribution:")
    for k, v in vr["label_distribution"].items():
        pct = round(100 * v / vr["total"], 1) if vr["total"] else 0
        print(f"  {k}: {v} ({pct}%)")
    print()
    print("Hallucination category distribution:")
    for k, v in sorted(vr["category_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()

    if vr["records"]:
        print(f"Issues found in {len(vr['records'])} records (first 10 shown):")
        for r in vr["records"][:10]:
            print(f"  [{r['status']}] {r['record_id']}")
            for e in r["errors"]:
                print(f"    ❌ ERROR: {e}")
            for w in r["warnings"]:
                print(f"    ⚠️  WARN:  {w}")
    else:
        print("✅ No issues found — dataset is TrustGuard-compatible!")

    if vr["compatibility_score"] >= 95:
        verdict = "✅ READY for TrustGuard pipeline"
    elif vr["compatibility_score"] >= 80:
        verdict = "⚠️  MOSTLY COMPATIBLE — fix errors before running pipeline"
    else:
        verdict = "❌ INCOMPATIBLE — significant conversion issues"

    print()
    print(f"Verdict: {verdict}")
    print("=" * 60)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vr, f, indent=2)
        print(f"Full report saved: {output_path}")


def show_template():
    print("""
TrustGuard-compatible record format (from CICIDS):

{
  "record_id":              "CICIDS-00001",
  "rule_id":                "R-A1B2C3D4",
  "action":                 "DENY",           // ALLOW | DENY | BLOCK | DROP
  "protocol":               "TCP",            // TCP | UDP | ICMP | ANY
  "src_ip":                 "192.168.1.100",
  "src_port":               "52341",
  "dst_ip":                 "10.0.0.5",
  "dst_port":               "80",
  "direction":              "inbound",        // inbound | outbound | both
  "intent_description":     "Block HTTP flood from infected host",
  "generated_rule":         "{\\"action\\":\\"DENY\\",\\"protocol\\":\\"TCP\\",...}",
  "hallucination_label":    "hallucinated",   // correct | hallucinated
  "hallucination_category": "scope_expansion",// or null for correct
  "confidence":             0.72,             // 0.0 – 1.0
  "reasoning":              "Flow classified as DDoS...",
  "priority":               1,
  "chain_of_thought":       "Step1: ...",
  "semantic_score":         0.81,
  "compliance_tags":        ["ZERO_TRUST"],
  "original_flow_label":    "DDoS",
  "source_dataset":         "CICIDS2017"
}

Hallucination category mapping from CICIDS labels:
  BENIGN              → label=correct,      category=null
  DDoS                → label=hallucinated, category=scope_expansion
  PortScan            → label=hallucinated, category=wrong_port
  Bot                 → label=hallucinated, category=intent_flip
  Web Attack (any)    → label=hallucinated, category=security_downgrade
  FTP/SSH-Patator     → label=hallucinated, category=wrong_protocol
  Heartbleed          → label=hallucinated, category=missing_constraint
  Infiltration        → label=hallucinated, category=over_permissive
""")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CICIDS → TrustGuard Converter & Validator")
    parser.add_argument("--mode", choices=["convert", "validate", "template"],
                        default="template", help="Operation mode")
    parser.add_argument("--input",  help="Input file (CSV for convert, JSON for validate)")
    parser.add_argument("--output", help="Output JSON path (for convert)")
    parser.add_argument("--report", help="Validation report output path (JSON)")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Max records to convert (default: all)")
    parser.add_argument("--use_ollama", action="store_true",
                        help="Enrich rules via Ollama (requires Ollama running)")
    parser.add_argument("--ollama_model", default="llama3.1:8b")
    parser.add_argument("--no_balance", action="store_true",
                        help="Skip class balancing")
    args = parser.parse_args()

    if args.mode == "template":
        show_template()
        return

    if not args.input:
        print("Error: --input required for this mode")
        sys.exit(1)

    if args.mode == "convert":
        out = args.output or args.input.replace(".csv", "_trustguard.json")
        records = convert_cicids(
            args.input, out,
            max_records=args.max_records,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model,
            balance_classes=not args.no_balance
        )
        print("\nRunning compatibility validation...")
        vr = validate_dataset(records)
        report_out = args.report or out.replace(".json", "_validation.json")
        print_validation_report(vr, report_out)

    elif args.mode == "validate":
        print(f"Loading: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            print("Error: JSON must contain a 'records' list or be a list directly")
            sys.exit(1)
        print(f"Validating {len(records)} records...")
        vr = validate_dataset(records)
        print_validation_report(vr, args.report)


if __name__ == "__main__":
    main()