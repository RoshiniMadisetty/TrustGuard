"""
TrustGuard - Person 3
Compliance Checker v2

Fixes over v1:
  - Context-aware rules (outbound web browsing is LOW not CRITICAL)
  - Proper severity levels: CRITICAL / HIGH / MEDIUM / LOW / INFO
  - Risk score contribution proportional to severity
  - Reviewer-ready justifications for every flag

Frameworks:
  - Least Privilege Principle
  - Zero Trust
  - PCI-DSS (1.2, 1.3, 2.2.1, 6.4)
  - HIPAA (164.312)
  - General Security Best Practices

Run: python compliance_checker_v2.py
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC


# ── Severity weights for risk score contribution ──────────────────────────────
SEVERITY_WEIGHTS = {
    "critical": 0.40,
    "high":     0.25,
    "medium":   0.12,
    "low":      0.05,
    "info":     0.01,
}

WILDCARD = {"any", "*", "0.0.0.0/0", "all", ""}


# ──────────────────────────────────────────────
# RESULT
# ──────────────────────────────────────────────

@dataclass
class ComplianceResult:
    compliant:               bool  = True
    violations:              list  = field(default_factory=list)
    warnings:                list  = field(default_factory=list)
    frameworks_checked:      list  = field(default_factory=list)
    risk_score_contribution: float = 0.0

    def add_violation(self, framework: str, rule_id: str,
                      message: str, severity: str, justification: str = ""):
        self.compliant = False
        self.violations.append({
            "framework":     framework,
            "rule_id":       rule_id,
            "message":       message,
            "severity":      severity,
            "justification": justification,
        })
        self.risk_score_contribution = min(
            1.0,
            self.risk_score_contribution + SEVERITY_WEIGHTS.get(severity, 0.05)
        )

    def add_warning(self, framework: str, rule_id: str,
                    message: str, justification: str = ""):
        self.warnings.append({
            "framework":     framework,
            "rule_id":       rule_id,
            "message":       message,
            "justification": justification,
        })

    def to_dict(self) -> dict:
        by_severity = {}
        for v in self.violations:
            s = v["severity"]
            by_severity[s] = by_severity.get(s, 0) + 1
        return {
            "compliant":                 self.compliant,
            "violations":                self.violations,
            "warnings":                  self.warnings,
            "frameworks_checked":        self.frameworks_checked,
            "violation_count":           len(self.violations),
            "warning_count":             len(self.warnings),
            "violations_by_severity":    by_severity,
            "risk_score_contribution":   round(self.risk_score_contribution, 3),
        }


# ──────────────────────────────────────────────
# COMPLIANCE CHECKER v2
# ──────────────────────────────────────────────

class ComplianceCheckerV3:

    def check(self, requirement: str,
              rule: dict | None) -> ComplianceResult:
        result = ComplianceResult()

        if rule is None:
            result.add_warning("GENERAL", "GEN-000",
                               "No rule to check — LLM failed to generate output")
            return result

        action   = str(rule.get("action",              "")).lower()
        protocol = str(rule.get("protocol",            "")).lower()
        source   = str(rule.get("source",         "any")).lower()
        dest     = str(rule.get("destination",    "any")).lower()
        dst_port = str(rule.get("destination_port","any")).lower()
        req_low  = requirement.lower()

        # Context flags — used to calibrate severity
        is_sensitive_env = any(kw in req_low for kw in [
            "cardholder", "payment", "card", "pci",
            "patient", "health", "medical", "hipaa", "phi",
            "financial", "payroll", "hr system"
        ])
        is_outbound_browsing = (
            action == "allow"
            and dst_port in {"443", "80", "any", "*"}
            and "web" in req_low or "browsing" in req_low or "internet" in req_low
        )
        is_internal_only = any(kw in source for kw in [
            "internal", "vlan", "10.", "192.168", "172.16",
            "employees", "corporate", "management"
        ])

        # ── 1. LEAST PRIVILEGE ────────────────────────────────────────────────
        result.frameworks_checked.append("Least Privilege Principle")

        if (action == "allow"
                and source in WILDCARD
                and dest in WILDCARD
                and dst_port in {"any", "*", "0"}):
            result.add_violation(
                "Least Privilege", "LPP-001",
                "allow any→any:any — completely unrestricted access",
                "critical",
                "Least Privilege requires that access be explicitly granted "
                "to specific sources, destinations, and ports only."
            )

        elif action == "allow" and source in WILDCARD and dest in WILDCARD:
            result.add_violation(
                "Least Privilege", "LPP-002",
                "allow any→any — both source and destination unrestricted",
                "high",
                "At minimum, either source or destination must be explicitly specified."
            )

        elif action == "allow" and source in WILDCARD and not is_internal_only:
            # Outbound web browsing for all users is MEDIUM not HIGH
            severity = "low" if is_outbound_browsing else "medium"
            result.add_violation(
                "Least Privilege", "LPP-003",
                f"allow from any source — source not constrained",
                severity,
                "Many organisations permit outbound web browsing for all users "
                "(LOW). However, inbound rules with 'any' source "
                "are a meaningful risk (MEDIUM/HIGH)."
                if is_outbound_browsing else
                "Source should be restricted to an explicit IP range or object."
            )

        # ── 2. ZERO TRUST ─────────────────────────────────────────────────────
        result.frameworks_checked.append("Zero Trust")

        if action == "allow" and source in WILDCARD and not is_outbound_browsing:
            result.add_violation(
                "Zero Trust", "ZT-001",
                "Source is 'any' — violates Zero Trust 'never trust, always verify'",
                "high" if is_sensitive_env else "medium",
                "Zero Trust mandates explicit identity verification for every "
                "access request. 'Any' source bypasses this entirely."
            )

        if (action == "allow"
                and dest in {"0.0.0.0/0", "any"}
                and dst_port in {"any", "*"}
                and source not in WILDCARD):
            result.add_violation(
                "Zero Trust", "ZT-002",
                "Unrestricted outbound from a named source — egress not controlled",
                "medium",
                "Zero Trust requires both ingress and egress to be explicitly "
                "controlled. Unrestricted outbound allows data exfiltration."
            )

        # ── 3. PCI-DSS ────────────────────────────────────────────────────────
        result.frameworks_checked.append("PCI-DSS")
        pci_keywords = ["cardholder", "payment", "card", "pci", "pos "]

        if any(kw in req_low for kw in pci_keywords):
            # PCI-DSS 1.3: No direct internet to CDE
            if action == "allow" and (source in WILDCARD or "internet" in req_low):
                result.add_violation(
                    "PCI-DSS", "PCI-DSS-1.3",
                    "Direct internet access to cardholder data environment prohibited",
                    "critical",
                    "PCI-DSS Requirement 1.3: Prohibit direct public access "
                    "between the internet and any component in the CDE."
                )
            # PCI-DSS 2.2.1: No insecure protocols
            if dst_port in {"23", "21", "80"} and action == "allow":
                names = {"23": "Telnet", "21": "FTP", "80": "HTTP"}
                result.add_violation(
                    "PCI-DSS", "PCI-DSS-2.2.1",
                    f"{names[dst_port]} is prohibited in CDE — use encrypted equivalent",
                    "critical",
                    "PCI-DSS 2.2.1: System configuration standards must include "
                    "removal of all unnecessary functionality and insecure protocols."
                )

        # PCI-DSS 1.2: Default deny (flag if rule is allow-all in likely CDE context)
        if (action == "allow"
                and source in WILDCARD
                and dest in WILDCARD
                and is_sensitive_env):
            result.add_violation(
                "PCI-DSS", "PCI-DSS-1.2",
                "Permissive rule in sensitive environment violates default-deny requirement",
                "critical",
                "PCI-DSS 1.2: Deny all traffic except that explicitly required."
            )

        # ── 4. HIPAA ──────────────────────────────────────────────────────────
        result.frameworks_checked.append("HIPAA")
        hipaa_keywords = ["patient", "health", "medical", "hipaa", "phi",
                          "ehr", "health record", "clinical"]

        if any(kw in req_low for kw in hipaa_keywords):
            if action == "allow" and source in WILDCARD:
                result.add_violation(
                    "HIPAA", "HIPAA-164.312(a)(1)",
                    "Access to PHI systems must be restricted to authorised users — "
                    "'any' source not permitted",
                    "critical",
                    "HIPAA §164.312(a)(1): Implement technical policies and "
                    "procedures for electronic information systems that allow access "
                    "only to authorised persons."
                )
            if dst_port == "80" and action == "allow":
                result.add_violation(
                    "HIPAA", "HIPAA-164.312(e)(2)(ii)",
                    "PHI must be transmitted over encrypted channels — "
                    "HTTP (port 80) is unencrypted",
                    "critical",
                    "HIPAA §164.312(e)(2)(ii): Implement encryption and decryption "
                    "mechanism for PHI in transit."
                )

        # ── 5. SECURITY BEST PRACTICES ────────────────────────────────────────
        result.frameworks_checked.append("Security Best Practices")

        # Telnet — always high, not critical (unless sensitive env)
        if action == "allow" and dst_port == "23":
            result.add_violation(
                "Security Best Practices", "SEC-001",
                "Telnet (port 23) transmits data in plaintext",
                "critical" if is_sensitive_env else "high",
                "Telnet provides no encryption or authentication. "
                "SSH (port 22) should be used instead in all environments."
            )

        # FTP — medium (not high, it's common in legacy environments)
        if action == "allow" and dst_port == "21":
            result.add_violation(
                "Security Best Practices", "SEC-002",
                "FTP (port 21) transmits credentials in plaintext",
                "medium",
                "FTP is a legacy protocol with no encryption. "
                "SFTP (port 22) or FTPS is preferred. "
                "Many organisations still run FTP on isolated networks — MEDIUM."
            )

        # RDP from any source — critical (ransomware vector)
        if action == "allow" and dst_port == "3389" and source in WILDCARD:
            result.add_violation(
                "Security Best Practices", "SEC-003",
                "RDP (port 3389) exposed to unrestricted source",
                "critical",
                "RDP exposed to the internet is the leading initial access vector "
                "for ransomware campaigns. Must be restricted to VPN/management IPs."
            )

        # SSH from any source — high (not critical, widely deployed)
        if action == "allow" and dst_port == "22" and source in WILDCARD:
            result.add_violation(
                "Security Best Practices", "SEC-004",
                "SSH (port 22) open to any source",
                "high",
                "SSH from any source exposes authentication to brute-force attacks. "
                "Restrict to management VLAN, VPN, or specific IPs."
            )

        # Dev → prod — critical
        if action == "allow":
            if ("dev" in source and
                    ("prod" in dest or "production" in dest)):
                result.add_violation(
                    "Security Best Practices", "SEC-005",
                    "Direct dev→production access allowed",
                    "critical",
                    "Dev-to-prod lateral movement bypasses change management, "
                    "testing, and audit controls. Use CI/CD pipelines."
                )

        # Outbound web browsing for all users — LOW (not a real violation)
        if is_outbound_browsing and action == "allow":
            result.add_warning(
                "Security Best Practices", "SEC-INFO-001",
                "Outbound web browsing for all users — common policy, review intent",
                "Many organisations explicitly permit outbound HTTPS/HTTP for users. "
                "This is LOW risk unless in a restricted environment (CDE, PHI). "
                "Consider URL filtering rather than port-level blocking."
            )

        return result


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_compliance_v2(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/compliance_v2_results.json"
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs   = data["pairs"]
    checker = ComplianceCheckerV3()

    results      = []
    compliant_ct = 0
    violation_ct = 0
    fw_violations = {}
    severity_counts = {}

    print(f"\n{'='*68}")
    print(f"TrustGuard — Compliance Checker v2")
    print(f"Frameworks: Least Privilege | Zero Trust | PCI-DSS | HIPAA | Best Practices")
    print(f"Checking {len(pairs)} rules...")
    print(f"{'='*68}\n")

    for pair in pairs:
        req  = pair["requirement"]
        rule = pair.get("generated_rule")
        cr   = checker.check(req, rule)

        if cr.compliant:
            compliant_ct += 1
            flag = "✓"
        else:
            violation_ct += 1
            flag = "✗"
            for v in cr.violations:
                fw = v["framework"]
                sv = v["severity"]
                fw_violations[fw]    = fw_violations.get(fw, 0) + 1
                severity_counts[sv]  = severity_counts.get(sv, 0) + 1

        print(f"  [{flag}] {pair['pair_id']} | {pair.get('label','?'):<12} | "
              f"risk:{cr.risk_score_contribution:.2f} | "
              f"{req[:45]}...")
        for v in cr.violations:
            sev_icon = {"critical":"🔴","high":"🟠","medium":"🟡",
                        "low":"🟢","info":"⚪"}.get(v["severity"],"⚪")
            print(f"       {sev_icon} [{v['rule_id']}] {v['severity'].upper()}: "
                  f"{v['message']}")
        for w in cr.warnings:
            print(f"       ℹ  [{w['rule_id']}] INFO: {w['message']}")

        results.append({
            "pair_id":               pair["pair_id"],
            "requirement":           req,
            "p1_label":              pair.get("label","unknown"),
            "p1_hallucination_type": pair.get("hallucination_type","none"),
            "compliance_result":     cr.to_dict()
        })

    total = len(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "created_at":             datetime.now(UTC).isoformat(),
            "total":                  total,
            "compliant":              compliant_ct,
            "non_compliant":          violation_ct,
            "compliance_rate":        f"{compliant_ct/total*100:.1f}%",
            "framework_violations":   fw_violations,
            "violations_by_severity": severity_counts,
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  ✓ Compliant:       {compliant_ct} ({compliant_ct/total*100:.1f}%)")
    print(f"  ✗ Non-compliant:   {violation_ct} ({violation_ct/total*100:.1f}%)")
    print(f"\n  Violations by severity:")
    for sv in ["critical","high","medium","low","info"]:
        cnt = severity_counts.get(sv, 0)
        if cnt:
            print(f"    {sv:<10} {cnt}")
    print(f"\n  Violations by framework:")
    for fw, cnt in sorted(fw_violations.items(), key=lambda x: -x[1]):
        print(f"    {fw:<35} {cnt}")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*68}\n")

    return output


if __name__ == "__main__":
    run_compliance_v2()