"""
TrustGuard - Person 3, Week 1
Syntax Validator

Rule-based (not vendor-specific yet) syntax checker.
Validates LLM-generated firewall rules against the JSON schema
and basic structural correctness rules.

Run: python syntax_validator.py
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# VALID VALUES
# ──────────────────────────────────────────────

VALID_ACTIONS    = {"allow", "deny", "drop", "reject"}
VALID_PROTOCOLS  = {"tcp", "udp", "icmp", "any", "http", "https",
                    "ssh", "rdp", "dns", "ftp", "smtp"}
VALID_DIRECTIONS = {"inbound", "outbound", "both"}

VALID_PORT_RANGE = range(0, 65536)   # 0–65535

# Known dangerous / suspicious field values
WILDCARD_VALUES  = {"any", "*", "0.0.0.0/0", "all", "0.0.0.0"}

# Regex patterns for valid IPs and CIDRs
IP_PATTERN   = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)
CIDR_PATTERN = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}/\d{1,2}$"
)


# ──────────────────────────────────────────────
# VALIDATION RESULT
# ──────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


# ──────────────────────────────────────────────
# CORE VALIDATOR
# ──────────────────────────────────────────────

class SyntaxValidator:
    """
    Validates firewall rules for structural correctness.
    Week 1: rule-based, not vendor-specific.
    """

    def validate(self, rule: Optional[dict]) -> ValidationResult:
        result = ValidationResult()

        # ── Null check ────────────────────────────────────────────────────────
        if rule is None:
            result.add_error("Rule is None — LLM failed to generate output")
            return result

        if not isinstance(rule, dict):
            result.add_error(f"Rule must be a dict, got {type(rule).__name__}")
            return result

        # ── Required fields ───────────────────────────────────────────────────
        required = ["action", "protocol", "source", "destination", "destination_port"]
        for field_name in required:
            if field_name not in rule:
                result.add_error(f"Missing required field: '{field_name}'")
            elif rule[field_name] is None or str(rule[field_name]).strip() == "":
                result.add_error(f"Field '{field_name}' is empty or null")

        if not result.is_valid:
            return result   # No point checking further if fields are missing

        # ── Action validation ─────────────────────────────────────────────────
        action = str(rule.get("action", "")).lower().strip()
        if action not in VALID_ACTIONS:
            result.add_error(
                f"Invalid action '{action}'. Must be one of: {sorted(VALID_ACTIONS)}"
            )

        # ── Protocol validation ───────────────────────────────────────────────
        protocol = str(rule.get("protocol", "")).lower().strip()
        if protocol not in VALID_PROTOCOLS:
            result.add_error(
                f"Invalid or unknown protocol '{protocol}'. "
                f"Valid: {sorted(VALID_PROTOCOLS)}"
            )

        # ── Direction validation (optional field) ─────────────────────────────
        direction = str(rule.get("direction", "both")).lower().strip()
        if direction not in VALID_DIRECTIONS:
            result.add_warning(
                f"Invalid direction '{direction}'. "
                f"Valid: {sorted(VALID_DIRECTIONS)}. Defaulting to 'both'."
            )

        # ── Port validation ───────────────────────────────────────────────────
        dst_port = rule.get("destination_port")
        port_issues = self._validate_port(dst_port, "destination_port")
        for issue in port_issues:
            result.add_error(issue)

        src_port = rule.get("source_port")
        if src_port is not None:
            port_issues = self._validate_port(src_port, "source_port")
            for issue in port_issues:
                result.add_warning(issue)

        # ── Source/destination format check ───────────────────────────────────
        source = str(rule.get("source", ""))
        dest   = str(rule.get("destination", ""))

        src_issues  = self._validate_address(source, "source")
        dest_issues = self._validate_address(dest, "destination")

        for w in src_issues:
            result.add_warning(w)
        for w in dest_issues:
            result.add_warning(w)

        # ── Logical sanity checks ─────────────────────────────────────────────
        if action == "allow":
            if (source.lower() in WILDCARD_VALUES
                    and dest.lower() in WILDCARD_VALUES
                    and str(dst_port).lower() in {"any", "*", "0"}):
                result.add_error(
                    "DANGEROUS: allow any→any:any — unrestricted access rule"
                )

        # ── Description warning ───────────────────────────────────────────────
        if not rule.get("description"):
            result.add_warning("No description provided — hard to audit later")

        return result

    def _validate_port(self, port_value, field_name: str) -> list:
        issues = []
        if port_value is None:
            return issues

        port_str = str(port_value).lower().strip()

        if port_str in {"any", "*", ""}:
            return issues   # Wildcard is valid

        # Handle comma-separated ports (e.g. "80,443")
        parts = [p.strip() for p in port_str.split(",")]
        for part in parts:
            # Handle range (e.g. "1024-65535")
            if "-" in part:
                bounds = part.split("-")
                if len(bounds) == 2:
                    try:
                        lo, hi = int(bounds[0]), int(bounds[1])
                        if lo not in VALID_PORT_RANGE or hi not in VALID_PORT_RANGE:
                            issues.append(
                                f"Port range {part} in '{field_name}' out of 0-65535"
                            )
                        if lo > hi:
                            issues.append(
                                f"Port range {part} in '{field_name}': "
                                f"lower bound > upper bound"
                            )
                    except ValueError:
                        issues.append(
                            f"Invalid port range '{part}' in '{field_name}'"
                        )
                continue

            try:
                port_int = int(part)
                if port_int not in VALID_PORT_RANGE:
                    issues.append(
                        f"Port {port_int} in '{field_name}' out of valid range 0-65535"
                    )
            except ValueError:
                issues.append(
                    f"Non-numeric port value '{part}' in '{field_name}'"
                )

        return issues

    def _validate_address(self, address: str, field_name: str) -> list:
        warnings = []
        addr = address.strip().lower()

        if addr in WILDCARD_VALUES or not addr:
            return warnings   # Wildcards are structurally valid

        # Check for comma-separated (multiple addresses — not standard)
        if "," in addr:
            warnings.append(
                f"'{field_name}' contains multiple values separated by commas. "
                f"Standard rules have one source/destination."
            )
            return warnings

        # Check for " or " (LLM sometimes outputs "x or y")
        if " or " in addr:
            warnings.append(
                f"'{field_name}' contains 'or' — LLM generated multiple options. "
                f"Should be a single address."
            )
            return warnings

        # If it looks like an IP or CIDR, validate format
        if re.match(r"^\d", addr):
            if not (IP_PATTERN.match(addr) or CIDR_PATTERN.match(addr)):
                warnings.append(
                    f"'{field_name}' value '{address}' looks like an IP/CIDR "
                    f"but has invalid format"
                )
            else:
                # Validate octet ranges
                octets = re.findall(r"\d+", addr.split("/")[0])
                for octet in octets:
                    if int(octet) > 255:
                        warnings.append(
                            f"'{field_name}' IP '{address}' has invalid octet: {octet}"
                        )

        return warnings


# ──────────────────────────────────────────────
# BATCH VALIDATOR
# ──────────────────────────────────────────────

def validate_dataset(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path: str = "../outputs/syntax_validation_results.json"
) -> dict:
    """
    Runs syntax validator on entire labeled dataset.
    Outputs per-pair validation results.
    """
    import os
    from datetime import datetime, UTC

    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    validator = SyntaxValidator()

    results = []
    stats = {"valid": 0, "invalid": 0, "warnings_only": 0}

    print(f"\n{'='*60}")
    print(f"TrustGuard Person 3 — Syntax Validator")
    print(f"Validating {len(pairs)} rules...")
    print(f"{'='*60}\n")

    for pair in pairs:
        rule = pair.get("generated_rule")
        vr = validator.validate(rule)

        if not vr.is_valid:
            stats["invalid"] += 1
            flag = "✗"
        elif vr.warnings:
            stats["warnings_only"] += 1
            flag = "⚠"
        else:
            stats["valid"] += 1
            flag = "✓"

        results.append({
            "pair_id": pair["pair_id"],
            "requirement": pair["requirement"],
            "existing_label": pair.get("label", "unknown"),
            "existing_hallucination_type": pair.get("hallucination_type", "none"),
            "syntax_valid": vr.is_valid,
            "syntax_errors": vr.errors,
            "syntax_warnings": vr.warnings,
            "validation_result": vr.to_dict()
        })

        print(f"  [{flag}] {pair['pair_id']} | {pair['requirement'][:55]}...")
        for e in vr.errors:
            print(f"       ERROR: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "total": len(results),
            "valid": stats["valid"],
            "invalid": stats["invalid"],
            "warnings_only": stats["warnings_only"]
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✓ Valid:         {stats['valid']}")
    print(f"  ⚠ Warnings only: {stats['warnings_only']}")
    print(f"  ✗ Invalid:       {stats['invalid']}")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    validate_dataset()