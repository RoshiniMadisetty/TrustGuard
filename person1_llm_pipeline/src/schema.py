"""
TrustGuard - Week 1
Firewall Rule Schema Definition
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Action(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    DROP = "drop"
    REJECT = "reject"


class Protocol(str, Enum):
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ANY = "any"
    HTTP = "http"
    HTTPS = "https"
    SSH = "ssh"
    RDP = "rdp"
    DNS = "dns"
    FTP = "ftp"
    SMTP = "smtp"


class Direction(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BOTH = "both"


@dataclass
class FirewallRule:
    """
    Structured firewall rule schema.
    Represents a vendor-agnostic firewall rule.
    """
    rule_id: str
    action: Action
    protocol: Protocol
    source: str                         # IP, subnet, object name, or "any"
    destination: str                    # IP, subnet, object name, or "any"
    source_port: Optional[str] = None   # port number, range, or "any"
    destination_port: Optional[str] = None
    direction: Direction = Direction.BOTH
    description: Optional[str] = None
    vendor: Optional[str] = None        # cisco, fortinet, palo_alto, generic

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "action": self.action.value,
            "protocol": self.protocol.value,
            "source": self.source,
            "destination": self.destination,
            "source_port": self.source_port or "any",
            "destination_port": self.destination_port or "any",
            "direction": self.direction.value,
            "description": self.description,
            "vendor": self.vendor or "generic"
        }


@dataclass
class RequirementRulePair:
    """
    A seed pair: natural language requirement → generated firewall rule.
    Used as the base unit for the benchmark dataset.
    """
    pair_id: str
    requirement: str            # Human-written security requirement
    generated_rule: dict        # Raw LLM output (JSON)
    structured_rule: Optional[dict] = None  # Parsed into FirewallRule schema
    raw_llm_output: Optional[str] = None    # Full LLM response text

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "requirement": self.requirement,
            "generated_rule": self.generated_rule,
            "structured_rule": self.structured_rule,
            "raw_llm_output": self.raw_llm_output
        }


# ──────────────────────────────────────────────
# 20 Seed Requirements (Week 1 dataset seeds)
# ──────────────────────────────────────────────
SEED_REQUIREMENTS = [
    # BASIC ACCESS CONTROL
    "Allow employees to access the company website over HTTPS.",
    "Block all inbound SSH connections from the internet.",
    "Allow the database server to communicate with the application server on port 5432.",
    "Deny all outbound traffic to known malicious IP ranges.",
    "Allow DNS queries from internal network to the corporate DNS server.",

    # REMOTE ACCESS
    "Allow VPN users to access internal HR systems.",
    "Block RDP access from any external IP address.",
    "Allow IT admins to SSH into servers only from the management VLAN.",
    "Allow remote workers to connect to the file server over port 445.",
    "Deny any traffic from guest WiFi to the internal corporate network.",

    # WEB AND APPLICATION TRAFFIC
    "Allow web servers to send outbound HTTPS requests to payment APIs.",
    "Block HTTP traffic and allow only HTTPS for all web communication.",
    "Allow the CI/CD pipeline server to pull code from GitHub over HTTPS.",
    "Deny direct internet access from the database server.",
    "Allow monitoring server to collect SNMP data from all network devices.",

    # SECURITY AND COMPLIANCE
    "Block all ICMP ping requests from external sources.",
    "Allow security scanning tools to probe internal servers on all ports.",
    "Deny any traffic on port 23 (Telnet) across the entire network.",
    "Allow email server to send outbound SMTP traffic on port 587.",
    "Block all traffic between the development environment and the production network.",
]