"""
TrustGuard - Person 3, Week 4
Edge Case Handler

Handles the four edge cases that basic validators miss:
  1. Contradictory rules (Allow SSH + Deny SSH in same policy)
  2. Hidden security downgrades (HTTPS → HTTP+HTTPS)
  3. Nonexistent network objects (made-up IPs, unknown services)
  4. Overly broad rules (100 specific rules collapsed to allow any)

Also produces the policy_sequence_validator for multi-rule reasoning.

Run: python edge_case_handler.py
"""

import json
import re
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum


class EdgeCaseType(str, Enum):
    CONTRADICTORY_RULE    = "contradictory_rule"
    SECURITY_DOWNGRADE    = "security_downgrade"
    NONEXISTENT_OBJECT    = "nonexistent_object"
    OVERLY_BROAD          = "overly_broad"
    MISSING_DEFAULT_DENY  = "missing_default_deny"
    REDUNDANT_RULE        = "redundant_rule"


@dataclass
class EdgeCaseResult:
    edge_cases_found: list = field(default_factory=list)
    clean: bool = True

    def add(self, case_type: EdgeCaseType, message: str,
            severity: str = "high", details: dict = None):
        self.clean = False
        self.edge_cases_found.append({
            "type":     case_type.value,
            "message":  message,
            "severity": severity,
            "details":  details or {}
        })

    def to_dict(self) -> dict:
        return {
            "clean":            self.clean,
            "edge_cases_found": self.edge_cases_found,
            "count":            len(self.edge_cases_found)
        }


# ──────────────────────────────────────────────
# KNOWN OBJECTS REGISTRY
# In production this would load from a network
# asset inventory. We define a baseline here.
# ──────────────────────────────────────────────

KNOWN_OBJECTS = {
    # Standard network objects
    "any", "all", "0.0.0.0/0", "*",
    # Common named objects in test dataset
    "web_server", "db_server", "app_server", "mail_server",
    "dns_server", "monitoring_server", "backup_server",
    "hr_systems", "file_server", "ldap_server", "time_server",
    "siem", "cicd_server", "waf", "jump_server",
    # Network ranges
    "10.0.0.0/8", "192.168.0.0/16", "172.16.0.0/12",
    "10.0.0.0/16", "192.168.1.0/24", "192.168.10.0/24",
    # Named groups / VLANs
    "management_vlan", "guest_wifi", "vpn_pool", "vpn_group",
    "finance_vlan", "accounting_vlan", "security_vlan",
    "user_vlan", "iot_vlan", "dmz_servers", "dmz",
    "internal_servers", "internal_network", "corporate_network",
    "dev_environment", "development", "production_network",
    "partner_network", "remote_workers", "employees",
    "scanner_tool", "forensics_workstation", "ir_team",
    "network_devices", "servers", "workstations",
    "container_cluster", "lambda_function", "api_gateway",
    "load_balancer", "analytics_server", "s3_bucket",
    "auth_service", "backup_service", "dev_machines",
    "payment_api_ips", "threat_intel_feeds",
    "cardholder_data_systems", "patient_records",
    "offsite_storage", "financial_systems",
    "infected_host", "compromised_server",
    # IPs that appear in dataset
    "192.168.50.100", "github.com", "0.0.0.0/0",
    "192.168.1.100", "10.0.0.100", "10.0.0.200/32",
    "8.8.8.8", "8.8.4.242", "185.199.108.143",
    "malicious_ip_range_1", "malicious_ip_range_2",
    "malicious_ip_range_3",
    # Special
    "server_ip", "vpn group",
}

KNOWN_SERVICES = {
    "tcp", "udp", "icmp", "any",
    "http", "https", "ssh", "rdp", "dns", "ftp", "smtp",
    "sftp", "telnet", "snmp", "ldap", "ntp", "syslog",
    "smb", "postgres", "mysql", "rdp",
    # ports as strings
    "22", "80", "443", "3389", "53", "25", "587",
    "23", "21", "161", "445", "5432", "3306", "389",
    "123", "514", "8080", "8443", "any", "*"
}

# Patterns that suggest made-up objects / commands
SUSPICIOUS_PATTERNS = [
    r"[A-Z_]{3,}_\d+$",              # ADMIN_VPN_3, SERVER_NODE_7
    r"enable_\w+_protocol",          # enable_super_secure_protocol
    r"ai_agent_protocol",            # hallucinated service names
    r"super_secure",
    r"\btest_\w+",
    r"new_\w+_service",
    r"custom_\w+_protocol",
]


# ──────────────────────────────────────────────
# EDGE CASE DETECTOR (single rule)
# ──────────────────────────────────────────────

class EdgeCaseDetector:

    def detect(self, requirement: str, rule: dict | None) -> EdgeCaseResult:
        result = EdgeCaseResult()

        if rule is None:
            return result

        req_low  = requirement.lower()
        action   = str(rule.get("action",   "")).lower()
        protocol = str(rule.get("protocol", "")).lower()
        source   = str(rule.get("source",   "any")).lower()
        dest     = str(rule.get("destination", "any")).lower()
        dst_port = str(rule.get("destination_port", "any")).lower()
        wildcard = {"any", "*", "0.0.0.0/0", "all", ""}

        # ── 1. Security Downgrade ─────────────────────────────────────────────
        # HTTPS-only → HTTP allowed
        if ("only https" in req_low or "https only" in req_low
                or "block http" in req_low):
            if "80" in dst_port or protocol == "http":
                result.add(
                    EdgeCaseType.SECURITY_DOWNGRADE,
                    "Requirement enforces HTTPS-only but rule allows HTTP (port 80). "
                    "Downgrade attack vector reintroduced.",
                    "high",
                    {"req_signal": "https only", "rule_port": dst_port}
                )

        # SSH required but Telnet port present
        if "ssh" in req_low and dst_port == "23":
            result.add(
                EdgeCaseType.SECURITY_DOWNGRADE,
                "Requirement mentions SSH but rule opens Telnet (port 23). "
                "Telnet is unencrypted.",
                "critical",
                {"expected_port": "22", "rule_port": "23"}
            )

        # SFTP required but FTP opened
        if "sftp" in req_low and dst_port == "21":
            result.add(
                EdgeCaseType.SECURITY_DOWNGRADE,
                "Requirement specifies SFTP but rule opens FTP (port 21). "
                "FTP transmits credentials in plaintext.",
                "critical",
                {"expected_port": "22", "rule_port": "21"}
            )

        # ── 2. Nonexistent Network Object ─────────────────────────────────────
        for addr_field, addr_value in [("source", source), ("destination", dest)]:
            if not addr_value or addr_value in {"any", "*", "0.0.0.0/0"}:
                continue

            # Skip valid IPs and CIDRs
            if re.match(r"^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$", addr_value):
                continue

            # Check if it's a known named object
            if addr_value not in KNOWN_OBJECTS:
                # Check for suspicious hallucination patterns
                is_suspicious = any(
                    re.search(p, addr_value, re.IGNORECASE)
                    for p in SUSPICIOUS_PATTERNS
                )
                if is_suspicious or len(addr_value) > 40:
                    result.add(
                        EdgeCaseType.NONEXISTENT_OBJECT,
                        f"'{addr_field}' value '{addr_value}' appears to be a "
                        f"hallucinated or nonexistent network object. "
                        f"Requires asset inventory verification.",
                        "high",
                        {"field": addr_field, "value": addr_value}
                    )

        # Check for hallucinated protocol/service names
        if protocol not in KNOWN_SERVICES:
            result.add(
                EdgeCaseType.NONEXISTENT_OBJECT,
                f"Protocol/service '{protocol}' is not a recognised standard. "
                f"LLM may have hallucinated a vendor command or service name.",
                "high",
                {"field": "protocol", "value": protocol}
            )

        # Check for comma-separated or 'or'-style values (LLM multi-object output)
        if " or " in source or " or " in dest:
            result.add(
                EdgeCaseType.NONEXISTENT_OBJECT,
                "Rule contains 'or' in source/destination — "
                "LLM generated multiple options instead of a single object. "
                "Requires disambiguation.",
                "medium",
                {"source": source, "destination": dest}
            )

        # ── 3. Overly Broad Rule ──────────────────────────────────────────────
        if (action == "allow"
                and source in wildcard
                and dest in wildcard
                and dst_port in {"any", "*", "0"}):
            result.add(
                EdgeCaseType.OVERLY_BROAD,
                "Allow any→any:any — completely unrestricted rule. "
                "This collapses all security policy into a single dangerous rule.",
                "critical",
                {"action": action, "source": source,
                 "destination": dest, "port": dst_port}
            )
        elif action == "allow" and source in wildcard and dest in wildcard:
            result.add(
                EdgeCaseType.OVERLY_BROAD,
                "Allow any→any with unrestricted source and destination. "
                "Apply least-privilege: restrict source or destination.",
                "high",
                {"source": source, "destination": dest}
            )
        elif action == "allow" and dst_port in {"any", "*"} and source in wildcard:
            result.add(
                EdgeCaseType.OVERLY_BROAD,
                "Allow any source on any port — "
                "port range should be restricted to the required service only.",
                "medium",
                {"source": source, "port": dst_port}
            )

        # ── 4. Comma-separated ports = overly broad ───────────────────────────
        if "," in dst_port and action == "allow":
            port_list = [p.strip() for p in dst_port.split(",")]
            if len(port_list) > 5:
                result.add(
                    EdgeCaseType.OVERLY_BROAD,
                    f"Rule opens {len(port_list)} ports simultaneously. "
                    "This is likely a hallucination — split into individual rules.",
                    "medium",
                    {"ports": port_list}
                )

        return result


# ──────────────────────────────────────────────
# POLICY SEQUENCE VALIDATOR
# Multi-rule: detects contradictions across a set of rules
# ──────────────────────────────────────────────

class PolicySequenceValidator:
    """
    Validates a set of rules for inter-rule conflicts.
    Example: allow SSH on port 22 + deny SSH on port 22 = contradiction.
    """

    def detect_contradictions(self, rules: list[dict]) -> list[dict]:
        """
        Returns list of contradictory pairs found in the ruleset.
        """
        contradictions = []

        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                r1, r2 = rules[i], rules[j]
                conflict = self._check_pair(r1, r2, i, j)
                if conflict:
                    contradictions.append(conflict)

        return contradictions

    def _check_pair(self, r1: dict, r2: dict,
                    idx1: int, idx2: int) -> dict | None:
        a1 = str(r1.get("action",   "")).lower()
        a2 = str(r2.get("action",   "")).lower()
        p1 = str(r1.get("protocol", "any")).lower()
        p2 = str(r2.get("protocol", "any")).lower()
        port1 = str(r1.get("destination_port", "any")).lower()
        port2 = str(r2.get("destination_port", "any")).lower()
        src1  = str(r1.get("source", "any")).lower()
        src2  = str(r2.get("source", "any")).lower()
        dst1  = str(r1.get("destination", "any")).lower()
        dst2  = str(r2.get("destination", "any")).lower()

        # Direct contradiction: same protocol+port but opposite actions
        action_conflict = (
            a1 in {"allow"} and a2 in {"deny","drop","reject"}
            or a1 in {"deny","drop","reject"} and a2 in {"allow"}
        )

        proto_match = (p1 == p2 or "any" in {p1, p2})
        port_match  = (port1 == port2 or "any" in {port1, port2})
        src_match   = (src1 == src2 or "any" in {src1, src2})
        dst_match   = (dst1 == dst2 or "any" in {dst1, dst2})

        if action_conflict and proto_match and port_match and src_match and dst_match:
            return {
                "type":    EdgeCaseType.CONTRADICTORY_RULE.value,
                "rule_1":  {"index": idx1, "rule": r1},
                "rule_2":  {"index": idx2, "rule": r2},
                "message": (
                    f"Contradictory rules: Rule {idx1+1} "
                    f"({a1.upper()} {p1.upper()} port {port1}) "
                    f"conflicts with Rule {idx2+1} "
                    f"({a2.upper()} {p2.upper()} port {port2}) "
                    f"on same traffic flow"
                ),
                "severity": "critical"
            }

        return None

    def check_default_deny(self, rules: list[dict]) -> bool:
        """Returns True if ruleset ends with a default-deny-all rule."""
        if not rules:
            return False
        last = rules[-1]
        return (
            str(last.get("action", "")).lower() in {"deny", "drop"}
            and str(last.get("source", "")).lower() in {"any", "*", "0.0.0.0/0"}
            and str(last.get("destination", "")).lower() in {"any", "*", "0.0.0.0/0"}
        )


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_edge_case_detection(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/edge_case_results.json"
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs    = data["pairs"]
    detector = EdgeCaseDetector()
    seq_val  = PolicySequenceValidator()

    results       = []
    clean_count   = 0
    flagged_count = 0
    type_counts   = {}

    print(f"\n{'='*65}")
    print(f"TrustGuard Person 3 — Edge Case Handler")
    print(f"Checking {len(pairs)} rules...")
    print(f"{'='*65}\n")

    all_rules = [p.get("generated_rule") for p in pairs if p.get("generated_rule")]

    # ── Contradiction check across all rules ──────────────────────────────────
    contradictions = seq_val.detect_contradictions(all_rules)
    has_default_deny = seq_val.check_default_deny(all_rules)

    print(f"  Policy-level checks:")
    print(f"    Contradictory rule pairs: {len(contradictions)}")
    print(f"    Default-deny at end:      {'✓ Yes' if has_default_deny else '✗ No'}")
    for c in contradictions[:5]:   # show first 5
        print(f"    🔴 {c['message']}")
    print()

    # ── Per-rule edge case detection ──────────────────────────────────────────
    for pair in pairs:
        req  = pair["requirement"]
        rule = pair.get("generated_rule")
        ec   = detector.detect(req, rule)

        if ec.clean:
            clean_count += 1
            flag = "✓"
        else:
            flagged_count += 1
            flag = "✗"
            for case in ec.edge_cases_found:
                ct = case["type"]
                type_counts[ct] = type_counts.get(ct, 0) + 1

        print(f"  [{flag}] {pair['pair_id']} | {pair.get('label','?'):<12} | "
              f"{req[:50]}...")
        for case in ec.edge_cases_found:
            print(f"       🔴 [{case['type']}] {case['severity'].upper()}: "
                  f"{case['message']}")

        results.append({
            "pair_id":               pair["pair_id"],
            "requirement":           req,
            "p1_label":              pair.get("label", "unknown"),
            "p1_hallucination_type": pair.get("hallucination_type","none"),
            "edge_case_result":      ec.to_dict()
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "metadata": {
            "created_at":              datetime.now(UTC).isoformat(),
            "total":                   len(results),
            "clean":                   clean_count,
            "flagged":                 flagged_count,
            "edge_case_type_counts":   type_counts,
            "policy_level": {
                "contradictory_pairs": len(contradictions),
                "has_default_deny":    has_default_deny,
                "contradictions":      contradictions[:10]
            }
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✓ Clean:    {clean_count}")
    print(f"  ✗ Flagged:  {flagged_count}")
    print(f"\n  Edge case breakdown:")
    for etype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:<30} {cnt}")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*65}\n")

    return output


if __name__ == "__main__":
    run_edge_case_detection()