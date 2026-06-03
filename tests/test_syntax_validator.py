"""
Tests for Week 1 — Syntax Validator
Run with: pytest tests/test_syntax_validator.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validator.syntax_validator import SyntaxValidator, ValidationResult


SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "schemas", "firewall_rule_schema.json"
)


@pytest.fixture
def validator():
    return SyntaxValidator(schema_path=SCHEMA_PATH)


# ────────────────────────────────────────────────────────────────
# VALID RULES — should pass
# ────────────────────────────────────────────────────────────────

class TestValidRules:

    def test_complete_valid_rule(self, validator):
        rule = {
            "rule_id": "test_001",
            "action": "deny",
            "protocol": "tcp",
            "source": {"ip": "any", "zone": "internet"},
            "destination": {"ip": "10.0.0.5", "port": 22},
            "direction": "inbound",
            "priority": 10,
            "description": "Block SSH from internet",
            "enabled": True,
        }
        result = validator.validate_rule(rule)
        assert result.is_valid, f"Expected valid, got errors: {[v.message for v in result.violations]}"

    def test_cidr_source_ip(self, validator):
        rule = {
            "rule_id": "test_002",
            "action": "allow",
            "protocol": "tcp",
            "source": {"ip": "192.168.1.0/24"},
            "destination": {"ip": "10.0.0.1", "port": 443},
            "priority": 20,
            "description": "CIDR source IP",
        }
        result = validator.validate_rule(rule)
        assert result.is_valid

    def test_port_range_valid(self, validator):
        rule = {
            "rule_id": "test_003",
            "action": "allow",
            "protocol": "tcp",
            "source": {"ip": "10.0.0.1"},
            "destination": {"ip": "10.0.0.2", "port": "8080-8090"},
            "priority": 30,
            "description": "Valid port range",
        }
        result = validator.validate_rule(rule)
        assert result.is_valid

    def test_any_protocol(self, validator):
        rule = {
            "rule_id": "test_004",
            "action": "deny",
            "protocol": "any",
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.1", "port": 3389},
            "priority": 5,
            "description": "Block RDP any protocol",
        }
        result = validator.validate_rule(rule)
        assert result.is_valid

    def test_udp_protocol(self, validator):
        rule = {
            "rule_id": "test_005",
            "action": "allow",
            "protocol": "udp",
            "source": {"ip": "10.0.0.2"},
            "destination": {"ip": "8.8.8.8", "port": 53},
            "priority": 30,
            "description": "Allow DNS",
        }
        result = validator.validate_rule(rule)
        assert result.is_valid


# ────────────────────────────────────────────────────────────────
# INVALID ACTION — bad enum value
# ────────────────────────────────────────────────────────────────

class TestInvalidAction:

    def test_wrong_action_permit(self, validator):
        rule = {
            "rule_id": "bad_action_001",
            "action": "permit",          # not in enum
            "protocol": "tcp",
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.1", "port": 80},
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid
        assert any("action" in v.field_path for v in result.violations)

    def test_wrong_action_accept(self, validator):
        rule = {
            "rule_id": "bad_action_002",
            "action": "accept",           # not in enum
            "protocol": "tcp",
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.1", "port": 80},
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid


# ────────────────────────────────────────────────────────────────
# INVALID IP
# ────────────────────────────────────────────────────────────────

class TestInvalidIP:

    def test_invalid_octet_999(self, validator):
        rule = {
            "rule_id": "bad_ip_001",
            "action": "allow",
            "protocol": "tcp",
            "source": {"ip": "999.168.1.1"},   # octet 999 invalid
            "destination": {"ip": "10.0.0.1", "port": 80},
            "priority": 10,
            "description": "Bad source IP",
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid

    def test_invalid_cidr_prefix(self, validator):
        rule = {
            "rule_id": "bad_ip_002",
            "action": "allow",
            "protocol": "tcp",
            "source": {"ip": "10.0.0.0/99"},   # /99 is invalid
            "destination": {"ip": "10.0.0.1", "port": 443},
            "priority": 10,
            "description": "Bad CIDR prefix",
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid

    def test_random_string_as_ip(self, validator):
        rule = {
            "rule_id": "bad_ip_003",
            "action": "deny",
            "protocol": "tcp",
            "source": {"ip": "not-an-ip"},
            "destination": {"ip": "10.0.0.1", "port": 22},
            "priority": 10,
            "description": "String as IP",
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid


# ────────────────────────────────────────────────────────────────
# INVALID PORT RANGE
# ────────────────────────────────────────────────────────────────

class TestInvalidPort:

    def test_reversed_port_range(self, validator):
        rule = {
            "rule_id": "bad_port_001",
            "action": "deny",
            "protocol": "tcp",
            "source": {"ip": "10.0.0.1"},
            "destination": {"ip": "10.0.0.2", "port": "9000-8000"},  # reversed
            "priority": 10,
            "description": "Reversed port range",
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid

    def test_port_integer_out_of_range(self, validator):
        rule = {
            "rule_id": "bad_port_002",
            "action": "allow",
            "protocol": "tcp",
            "source": {"ip": "10.0.0.1"},
            "destination": {"ip": "10.0.0.2", "port": 99999},  # > 65535
            "priority": 10,
            "description": "Port too high",
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid


# ────────────────────────────────────────────────────────────────
# MISSING REQUIRED FIELDS
# ────────────────────────────────────────────────────────────────

class TestMissingFields:

    def test_missing_protocol(self, validator):
        rule = {
            "rule_id": "missing_001",
            "action": "deny",
            # protocol missing
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.5", "port": 22},
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid

    def test_missing_action(self, validator):
        rule = {
            "rule_id": "missing_002",
            # action missing
            "protocol": "tcp",
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.5", "port": 22},
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid

    def test_missing_source_ip(self, validator):
        rule = {
            "rule_id": "missing_003",
            "action": "deny",
            "protocol": "tcp",
            "source": {"zone": "internet"},   # ip missing
            "destination": {"ip": "10.0.0.5", "port": 22},
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid


# ────────────────────────────────────────────────────────────────
# INVALID PROTOCOL
# ────────────────────────────────────────────────────────────────

class TestInvalidProtocol:

    def test_ftp_as_protocol(self, validator):
        rule = {
            "rule_id": "bad_proto_001",
            "action": "allow",
            "protocol": "ftp",          # not in enum
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.5", "port": 21},
        }
        result = validator.validate_rule(rule)
        assert not result.is_valid


# ────────────────────────────────────────────────────────────────
# WARNINGS (valid rules but missing best-practice fields)
# ────────────────────────────────────────────────────────────────

class TestWarnings:

    def test_missing_priority_gives_warning(self, validator):
        rule = {
            "rule_id": "warn_001",
            "action": "allow",
            "protocol": "tcp",
            "source": {"ip": "10.0.0.1"},
            "destination": {"ip": "10.0.0.2", "port": 443},
            "description": "Has description but no priority",
        }
        result = validator.validate_rule(rule)
        assert result.is_valid           # still valid (warnings only)
        assert result.warning_count > 0
        assert any("priority" in v.field_path for v in result.violations)

    def test_missing_description_gives_warning(self, validator):
        rule = {
            "rule_id": "warn_002",
            "action": "deny",
            "protocol": "tcp",
            "source": {"ip": "any"},
            "destination": {"ip": "10.0.0.1", "port": 22},
            "priority": 10,
        }
        result = validator.validate_rule(rule)
        assert result.is_valid
        assert any("description" in v.field_path for v in result.violations)


# ────────────────────────────────────────────────────────────────
# BATCH VALIDATION
# ────────────────────────────────────────────────────────────────

class TestBatch:

    def test_batch_returns_correct_counts(self, validator):
        rules = [
            {  # valid
                "rule_id": "b001",
                "action": "deny",
                "protocol": "tcp",
                "source": {"ip": "any"},
                "destination": {"ip": "10.0.0.1", "port": 22},
                "priority": 1,
                "description": "Block SSH",
            },
            {  # invalid action
                "rule_id": "b002",
                "action": "permit",
                "protocol": "tcp",
                "source": {"ip": "any"},
                "destination": {"ip": "10.0.0.1", "port": 80},
            },
        ]
        results = validator.validate_batch(rules)
        assert len(results) == 2
        assert results[0].is_valid is True
        assert results[1].is_valid is False

    def test_seed_dataset_loads_and_runs(self, validator):
        """Run validator on the full 20-rule seed dataset — no crashes."""
        import json
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "seed_rules.json"
        )
        with open(data_path) as f:
            rules = json.load(f)

        results = validator.validate_batch(rules)
        assert len(results) == 20

        valid_ids = [r.rule_id for r in results if r.is_valid]
        invalid_ids = [r.rule_id for r in results if not r.is_valid]

        print(f"\n  Valid:   {len(valid_ids)}")
        print(f"  Invalid: {len(invalid_ids)}")
        print(f"  Invalid IDs: {invalid_ids}")

        # At least the intentionally bad rules must fail
        for bad_id in ["rule_bad_001", "rule_bad_002", "rule_bad_003",
                       "rule_bad_004", "rule_bad_005", "rule_bad_006"]:
            assert bad_id in invalid_ids, f"{bad_id} should be invalid"
