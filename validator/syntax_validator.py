"""
Week 1 — Syntax Validator
Validates LLM-generated firewall rules against the defined JSON schema.
Also runs additional regex + logic checks beyond schema validation.
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import Any
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator


# ─── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class ValidationViolation:
    """Represents a single syntax violation found in a rule."""
    rule_id: str
    violation_type: str       # e.g. "MISSING_FIELD", "INVALID_IP", "BAD_PORT"
    field_path: str           # e.g. "source.ip", "action"
    message: str
    severity: str             # "ERROR" or "WARNING"
    raw_value: Any = None


@dataclass
class ValidationResult:
    """Full result of validating one firewall rule."""
    rule_id: str
    is_valid: bool
    violations: list[ValidationViolation] = field(default_factory=list)
    warnings: list[ValidationViolation] = field(default_factory=list)

    @property
    def error_count(self):
        return len([v for v in self.violations if v.severity == "ERROR"])

    @property
    def warning_count(self):
        return len([v for v in self.violations if v.severity == "WARNING"])

    def summary(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"{status} | rule_id={self.rule_id} | "
            f"errors={self.error_count} | warnings={self.warning_count}"
        )


# ─── Regex Patterns ──────────────────────────────────────────────────────────

IP_PATTERN         = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
CIDR_PATTERN       = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$")
PORT_RANGE_PATTERN = re.compile(r"^\d{1,5}-\d{1,5}$")


# ─── Main Validator Class ─────────────────────────────────────────────────────

class SyntaxValidator:
    """
    Validates firewall rules against:
    1. JSON Schema (structure, types, enums)
    2. Custom regex checks (IP octets, port ranges)
    3. Logic checks (port range direction, CIDR prefix)
    """

    def __init__(self, schema_path: str = None):
        if schema_path is None:
            # Default: look next to this file
            base = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(base, "..", "schemas", "firewall_rule_schema.json")

        with open(schema_path, "r") as f:
            self.schema = json.load(f)

        self.json_validator = Draft7Validator(self.schema)

    # ── Public API ──────────────────────────────────────────────────────────

    def validate_rule(self, rule: dict) -> ValidationResult:
        """Validate a single rule dict. Returns a ValidationResult."""
        rule_id = rule.get("rule_id", "<unknown>")
        violations = []

        # Step 1: JSON Schema validation
        schema_errors = self._run_schema_validation(rule, rule_id)
        violations.extend(schema_errors)

        # Step 2: Extra regex + logic checks (only if schema passed basics)
        if not schema_errors:
            extra = self._run_extra_checks(rule, rule_id)
            violations.extend(extra)

        errors   = [v for v in violations if v.severity == "ERROR"]
        warnings = [v for v in violations if v.severity == "WARNING"]
        is_valid = len(errors) == 0

        return ValidationResult(
            rule_id=rule_id,
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
        )

    def validate_batch(self, rules: list[dict]) -> list[ValidationResult]:
        """Validate a list of rules. Returns list of ValidationResults."""
        return [self.validate_rule(r) for r in rules]

    # ── Schema Validation ───────────────────────────────────────────────────

    def _run_schema_validation(self, rule: dict, rule_id: str) -> list[ValidationViolation]:
        violations = []
        for error in self.json_validator.iter_errors(rule):
            path = ".".join(str(p) for p in error.absolute_path) or "root"
            violations.append(ValidationViolation(
                rule_id=rule_id,
                violation_type="SCHEMA_ERROR",
                field_path=path,
                message=error.message,
                severity="ERROR",
                raw_value=error.instance,
            ))
        return violations

    # ── Extra Regex + Logic Checks ──────────────────────────────────────────

    def _run_extra_checks(self, rule: dict, rule_id: str) -> list[ValidationViolation]:
        violations = []

        for side in ("source", "destination"):
            block = rule.get(side, {})
            ip_val = block.get("ip", "")

            if ip_val != "any":
                violations += self._check_ip(ip_val, rule_id, f"{side}.ip")

            port_val = block.get("port")
            if port_val is not None:
                violations += self._check_port(port_val, rule_id, f"{side}.port")

        violations += self._check_priority_warning(rule, rule_id)
        violations += self._check_description_warning(rule, rule_id)

        return violations

    def _check_ip(self, ip_str: str, rule_id: str, path: str) -> list[ValidationViolation]:
        violations = []

        if CIDR_PATTERN.match(ip_str):
            # Validate each octet 0–255 and prefix 0–32
            parts = ip_str.replace("/", ".").split(".")
            octets = parts[:4]
            prefix = int(parts[4])
            for o in octets:
                if not (0 <= int(o) <= 255):
                    violations.append(ValidationViolation(
                        rule_id=rule_id,
                        violation_type="INVALID_IP_OCTET",
                        field_path=path,
                        message=f"IP octet out of range (0-255): '{o}' in '{ip_str}'",
                        severity="ERROR",
                        raw_value=ip_str,
                    ))
            if not (0 <= prefix <= 32):
                violations.append(ValidationViolation(
                    rule_id=rule_id,
                    violation_type="INVALID_CIDR_PREFIX",
                    field_path=path,
                    message=f"CIDR prefix must be 0–32, got {prefix}",
                    severity="ERROR",
                    raw_value=ip_str,
                ))

        elif IP_PATTERN.match(ip_str):
            for o in ip_str.split("."):
                if not (0 <= int(o) <= 255):
                    violations.append(ValidationViolation(
                        rule_id=rule_id,
                        violation_type="INVALID_IP_OCTET",
                        field_path=path,
                        message=f"IP octet out of range: '{o}' in '{ip_str}'",
                        severity="ERROR",
                        raw_value=ip_str,
                    ))
        else:
            violations.append(ValidationViolation(
                rule_id=rule_id,
                violation_type="INVALID_IP_FORMAT",
                field_path=path,
                message=f"'{ip_str}' is not a valid IP, CIDR, or 'any'",
                severity="ERROR",
                raw_value=ip_str,
            ))

        return violations

    def _check_port(self, port_val, rule_id: str, path: str) -> list[ValidationViolation]:
        violations = []

        if isinstance(port_val, str) and port_val != "any":
            if PORT_RANGE_PATTERN.match(port_val):
                lo, hi = map(int, port_val.split("-"))
                if lo >= hi:
                    violations.append(ValidationViolation(
                        rule_id=rule_id,
                        violation_type="INVALID_PORT_RANGE",
                        field_path=path,
                        message=f"Port range start ({lo}) must be less than end ({hi})",
                        severity="ERROR",
                        raw_value=port_val,
                    ))
                if hi > 65535:
                    violations.append(ValidationViolation(
                        rule_id=rule_id,
                        violation_type="PORT_OUT_OF_RANGE",
                        field_path=path,
                        message=f"Port {hi} exceeds maximum 65535",
                        severity="ERROR",
                        raw_value=port_val,
                    ))
        return violations

    def _check_priority_warning(self, rule: dict, rule_id: str) -> list[ValidationViolation]:
        if "priority" not in rule:
            return [ValidationViolation(
                rule_id=rule_id,
                violation_type="MISSING_PRIORITY",
                field_path="priority",
                message="Rule has no 'priority' field — rule ordering may be ambiguous",
                severity="WARNING",
            )]
        return []

    def _check_description_warning(self, rule: dict, rule_id: str) -> list[ValidationViolation]:
        if not rule.get("description"):
            return [ValidationViolation(
                rule_id=rule_id,
                violation_type="MISSING_DESCRIPTION",
                field_path="description",
                message="No description provided — hard to audit rule intent",
                severity="WARNING",
            )]
        return []


# ─── Report Printer ──────────────────────────────────────────────────────────

def print_report(results: list[ValidationResult]) -> None:
    """Pretty-print validation results to the terminal."""
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
        GREEN  = Fore.GREEN
        RED    = Fore.RED
        YELLOW = Fore.YELLOW
        BOLD   = Style.BRIGHT
        RESET  = Style.RESET_ALL
    except ImportError:
        GREEN = RED = YELLOW = BOLD = RESET = ""

    total   = len(results)
    passed  = sum(1 for r in results if r.is_valid)
    failed  = total - passed

    print(f"\n{'='*60}")
    print(f"  SYNTAX VALIDATION REPORT")
    print(f"  Total: {total}  |  Passed: {GREEN}{passed}{RESET}  |  Failed: {RED}{failed}{RESET}")
    print(f"{'='*60}\n")

    for result in results:
        color = GREEN if result.is_valid else RED
        print(f"  {color}{result.summary()}{RESET}")
        for v in result.violations:
            if v.severity == "ERROR":
                print(f"    {RED}[ERROR]{RESET}   {v.field_path}: {v.message}")
            else:
                print(f"    {YELLOW}[WARN]{RESET}    {v.field_path}: {v.message}")
        print()

    print(f"{'='*60}\n")
