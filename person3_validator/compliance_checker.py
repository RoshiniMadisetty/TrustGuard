"""
TrustGuard - Person 3, Week 3
Compliance Checker

Checks generated firewall rules against:
  - Least Privilege Principle
  - Zero Trust Principles
  - PCI-DSS (Payment Card Industry)
  - HIPAA (Health data)
  - General security best practices

Run: python compliance_checker.py
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, UTC


# ──────────────────────────────────────────────
# COMPLIANCE RESULT
# ──────────────────────────────────────────────

@dataclass
class ComplianceResult:
    compliant: bool = True
    violations: list = field(default_factory=list)
    warnings: list  = field(default_factory=list)
    frameworks_checked: list = field(default_factory=list)
    risk_score_contribution: float = 0.0  # fed into risk engine later

    def add_violation(self, framework: str, rule_id: str,
                      message: str, severity: str = "high"):
        self.compliant = False
        self.violations.append({
            "framework":  framework,
            "rule_id":    rule_id,
            "message":    message,
            "severity":   severity
        })
        # Contribute to risk score based on severity
        weights = {"critical": 0.4, "high": 0.25, "medium": 0.15, "low": 0.05}
        self.risk_score_contribution += weights.get(severity, 0.1)

    def add_warning(self, framework: str, message: str):
        self.warnings.append({"framework": framework, "message": message})

    def to_dict(self) -> dict:
        return {
            "compliant":                 self.compliant,
            "violations":                self.violations,
            "warnings":                  self.warnings,
            "frameworks_checked":        self.frameworks_checked,
            "violation_count":           len(self.violations),
            "warning_count":             len(self.warnings),
            "risk_score_contribution":   round(min(self.risk_score_contribution, 1.0), 3)
        }


# ──────────────────────────────────────────────
# COMPLIANCE CHECKER
# ──────────────────────────────────────────────

class ComplianceChecker:

    def check(self, requirement: str, rule: dict | None) -> ComplianceResult:
        result = ComplianceResult()

        if rule is None:
            result.add_warning("GENERAL", "No rule to check — LLM failed to generate")
            return result

        action   = str(rule.get("action",   "")).lower()
        protocol = str(rule.get("protocol", "")).lower()
        source   = str(rule.get("source",   "any")).lower()
        dest     = str(rule.get("destination", "any")).lower()
        dst_port = str(rule.get("destination_port", "any")).lower()
        req_low  = requirement.lower()

        wildcard = {"any", "*", "0.0.0.0/0", "all", ""}

        # ── 1. LEAST PRIVILEGE ────────────────────────────────────────────────
        result.frameworks_checked.append("Least Privilege Principle")

        if (action == "allow"
                and source in wildcard
                and dest in wildcard
                and dst_port in {"any", "*", "0"}):
            result.add_violation(
                "Least Privilege",
                "LPP-001",
                "Rule grants unrestricted allow any→any:any — "
                "violates least privilege: access should be explicit and minimal",
                "critical"
            )

        if action == "allow" and source in wildcard and dest in wildcard:
            result.add_violation(
                "Least Privilege",
                "LPP-002",
                "Rule allows traffic from any source to any destination — "
                "source or destination must be restricted",
                "high"
            )

        if action == "allow" and dst_port in {"any", "*"} and source in wildcard:
            result.add_violation(
                "Least Privilege",
                "LPP-003",
                "Rule allows any port from any source — "
                "port range should be restricted to necessary services only",
                "high"
            )

        # ── 2. ZERO TRUST ─────────────────────────────────────────────────────
        result.frameworks_checked.append("Zero Trust")

        if action == "allow" and source in wildcard:
            result.add_violation(
                "Zero Trust",
                "ZT-001",
                "Zero Trust requires explicit source identity — "
                "'any' source violates 'never trust, always verify'",
                "high"
            )

        if action == "allow" and "internal" not in source and source not in wildcard:
            # Check if rule gives internal resources wide access to internet
            if dest in {"0.0.0.0/0", "any"} and dst_port in {"any", "*"}:
                result.add_violation(
                    "Zero Trust",
                    "ZT-002",
                    "Rule allows unrestricted outbound access — "
                    "Zero Trust requires explicit egress control",
                    "medium"
                )

        # ── 3. PCI-DSS ────────────────────────────────────────────────────────
        result.frameworks_checked.append("PCI-DSS")

        # PCI-DSS 1.3: No direct internet to cardholder data
        pci_sensitive = ["cardholder", "payment", "card", "pci", "pos ", "point of sale"]
        if any(kw in req_low for kw in pci_sensitive):
            if action == "allow" and (source in wildcard or "internet" in req_low):
                result.add_violation(
                    "PCI-DSS",
                    "PCI-1.3",
                    "PCI-DSS 1.3: Direct internet access to cardholder data "
                    "systems is prohibited",
                    "critical"
                )

        # PCI-DSS 2.2.1: No insecure protocols in cardholder environment
        insecure_ports = {"23", "21", "80"}
        if any(kw in req_low for kw in pci_sensitive):
            if dst_port in insecure_ports:
                proto_name = {"23": "Telnet", "21": "FTP", "80": "HTTP"}.get(dst_port)
                result.add_violation(
                    "PCI-DSS",
                    "PCI-2.2.1",
                    f"PCI-DSS 2.2.1: {proto_name} (port {dst_port}) is insecure "
                    f"and must not be used in cardholder data environments",
                    "critical"
                )

        # PCI-DSS 1.2: Deny all except explicitly required
        if action == "allow" and dest in wildcard and source in wildcard:
            result.add_violation(
                "PCI-DSS",
                "PCI-1.2",
                "PCI-DSS 1.2: Firewall must deny all traffic except "
                "explicitly required — this rule is too permissive",
                "high"
            )

        # ── 4. HIPAA ──────────────────────────────────────────────────────────
        result.frameworks_checked.append("HIPAA")

        hipaa_sensitive = ["patient", "health", "medical", "hipaa", "phi",
                           "ehr", "health record", "clinical"]
        if any(kw in req_low for kw in hipaa_sensitive):
            if action == "allow" and source in wildcard:
                result.add_violation(
                    "HIPAA",
                    "HIPAA-164.312(a)(1)",
                    "HIPAA §164.312(a)(1): Access to PHI systems must be restricted "
                    "to authorised users only — 'any' source is not permitted",
                    "critical"
                )

            if dst_port == "80":
                result.add_violation(
                    "HIPAA",
                    "HIPAA-164.312(e)(2)(ii)",
                    "HIPAA §164.312(e)(2)(ii): PHI must be transmitted over "
                    "encrypted channels — HTTP (port 80) is not encrypted",
                    "critical"
                )

        # ── 5. GENERAL SECURITY BEST PRACTICES ───────────────────────────────
        result.frameworks_checked.append("Security Best Practices")

        # Telnet is always a violation
        if action == "allow" and dst_port == "23":
            result.add_violation(
                "Security Best Practices",
                "SEC-001",
                "Telnet (port 23) transmits data in plaintext — "
                "use SSH (port 22) instead",
                "high"
            )

        # FTP is a violation if SFTP available
        if action == "allow" and dst_port == "21":
            result.add_violation(
                "Security Best Practices",
                "SEC-002",
                "FTP (port 21) is insecure — "
                "use SFTP (port 22) or FTPS instead",
                "medium"
            )

        # RDP from internet = always critical
        if action == "allow" and dst_port == "3389" and source in wildcard:
            result.add_violation(
                "Security Best Practices",
                "SEC-003",
                "RDP (port 3389) exposed to any source — "
                "RDP from internet is a leading attack vector (ransomware)",
                "critical"
            )

        # SSH from internet with any source
        if action == "allow" and dst_port == "22" and source in wildcard:
            result.add_violation(
                "Security Best Practices",
                "SEC-004",
                "SSH (port 22) open to any source — "
                "restrict SSH to specific management IPs or VPN only",
                "high"
            )

        # Dev to prod is always prohibited
        if action == "allow":
            if "dev" in source and ("prod" in dest or "production" in dest):
                result.add_violation(
                    "Security Best Practices",
                    "SEC-005",
                    "Direct dev→production access is prohibited — "
                    "use CI/CD pipelines with controlled promotion",
                    "critical"
                )

        # Overly broad outbound from sensitive systems
        sensitive_sources = ["db_server", "database", "cardholder", "payment"]
        if any(s in source for s in sensitive_sources):
            if action == "allow" and dest in wildcard:
                result.add_violation(
                    "Security Best Practices",
                    "SEC-006",
                    f"Sensitive system '{source}' has unrestricted outbound access — "
                    "egress should be tightly controlled",
                    "high"
                )

        return result


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_compliance_check(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/compliance_results.json"
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs   = data["pairs"]
    checker = ComplianceChecker()

    results       = []
    compliant_ct  = 0
    violation_ct  = 0
    framework_violations = {}

    print(f"\n{'='*65}")
    print(f"TrustGuard Person 3 — Compliance Checker")
    print(f"Frameworks: Least Privilege, Zero Trust, PCI-DSS, HIPAA, Best Practices")
    print(f"Checking {len(pairs)} rules...")
    print(f"{'='*65}\n")

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
                framework_violations[fw] = framework_violations.get(fw, 0) + 1

        print(f"  [{flag}] {pair['pair_id']} | {pair.get('label','?'):<12} | "
              f"risk_contrib:{cr.risk_score_contribution:.2f} | "
              f"{req[:45]}...")
        for v in cr.violations:
            print(f"       🔴 [{v['framework']}][{v['rule_id']}] "
                  f"{v['severity'].upper()}: {v['message']}")
        for w in cr.warnings:
            print(f"       ⚠  [{w['framework']}] {w['message']}")

        results.append({
            "pair_id":               pair["pair_id"],
            "requirement":           req,
            "p1_label":              pair.get("label","unknown"),
            "p1_hallucination_type": pair.get("hallucination_type","none"),
            "compliance_result":     cr.to_dict()
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "metadata": {
            "created_at":            datetime.now(UTC).isoformat(),
            "total":                 len(results),
            "compliant":             compliant_ct,
            "non_compliant":         violation_ct,
            "compliance_rate":       f"{compliant_ct/len(results)*100:.1f}%",
            "framework_violations":  framework_violations
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✓ Compliant:     {compliant_ct}")
    print(f"  ✗ Non-compliant: {violation_ct} "
          f"({violation_ct/len(results)*100:.1f}%)")
    print(f"\n  Violations by framework:")
    for fw, cnt in sorted(framework_violations.items(), key=lambda x: -x[1]):
        print(f"    {fw:<35} {cnt} violations")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*65}\n")

    return output


if __name__ == "__main__":
    run_compliance_check()