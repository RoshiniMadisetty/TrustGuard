"""
TrustGuard - Week 1
Mock LLM Runner (No API Key Required)

This simulates what the real LLM pipeline produces,
so you can test Week 1 and Week 2 code immediately.

Run: python mock_runner.py
"""

import json
import os
from datetime import datetime

# Simulated LLM outputs for 20 seed requirements
# Mix of correct rules and hallucinated/dangerous ones (realistic simulation)
MOCK_LLM_OUTPUTS = [
    # 1. Allow employees to access company website over HTTPS
    {
        "action": "allow", "protocol": "tcp", "source": "10.0.0.0/8",
        "destination": "web_server", "source_port": "any",
        "destination_port": "443", "direction": "outbound",
        "description": "Allow internal employees to access company website over HTTPS"
    },
    # 2. Block all inbound SSH — HALLUCINATED: allows instead of blocks
    {
        "action": "allow", "protocol": "tcp", "source": "any",
        "destination": "any", "source_port": "any",
        "destination_port": "22", "direction": "inbound",
        "description": "SSH access rule"
    },
    # 3. DB ↔ App server on 5432
    {
        "action": "allow", "protocol": "tcp", "source": "app_server",
        "destination": "db_server", "source_port": "any",
        "destination_port": "5432", "direction": "both",
        "description": "Allow app server to communicate with database on PostgreSQL port"
    },
    # 4. Deny outbound to malicious IPs — HALLUCINATED: too broad
    {
        "action": "deny", "protocol": "any", "source": "any",
        "destination": "any", "source_port": "any",
        "destination_port": "any", "direction": "outbound",
        "description": "Block all outbound traffic"
    },
    # 5. DNS to corporate DNS server
    {
        "action": "allow", "protocol": "udp", "source": "10.0.0.0/8",
        "destination": "dns_server", "source_port": "any",
        "destination_port": "53", "direction": "outbound",
        "description": "Allow DNS queries from internal network to corporate DNS"
    },
    # 6. VPN users → HR systems
    {
        "action": "allow", "protocol": "tcp", "source": "vpn_pool",
        "destination": "hr_systems", "source_port": "any",
        "destination_port": "443", "direction": "inbound",
        "description": "Allow VPN users to access HR systems"
    },
    # 7. Block RDP from external — CORRECT
    {
        "action": "deny", "protocol": "tcp", "source": "any",
        "destination": "any", "source_port": "any",
        "destination_port": "3389", "direction": "inbound",
        "description": "Block RDP access from external IP addresses"
    },
    # 8. IT admins SSH from mgmt VLAN
    {
        "action": "allow", "protocol": "tcp", "source": "management_vlan",
        "destination": "servers", "source_port": "any",
        "destination_port": "22", "direction": "inbound",
        "description": "Allow IT admins to SSH into servers from management VLAN only"
    },
    # 9. Remote workers → file server 445 — HALLUCINATED: uses wrong port
    {
        "action": "allow", "protocol": "tcp", "source": "remote_workers",
        "destination": "file_server", "source_port": "any",
        "destination_port": "80", "direction": "inbound",
        "description": "Allow remote workers to connect to file server"
    },
    # 10. Deny guest WiFi → corporate
    {
        "action": "deny", "protocol": "any", "source": "guest_wifi",
        "destination": "10.0.0.0/8", "source_port": "any",
        "destination_port": "any", "direction": "both",
        "description": "Deny all traffic from guest WiFi to internal corporate network"
    },
    # 11. Web servers → payment APIs outbound HTTPS
    {
        "action": "allow", "protocol": "tcp", "source": "web_servers",
        "destination": "payment_api_ips", "source_port": "any",
        "destination_port": "443", "direction": "outbound",
        "description": "Allow web servers to make outbound HTTPS calls to payment APIs"
    },
    # 12. Block HTTP allow only HTTPS — HALLUCINATED: allows both
    {
        "action": "allow", "protocol": "tcp", "source": "any",
        "destination": "any", "source_port": "any",
        "destination_port": "80,443", "direction": "both",
        "description": "Allow web traffic on ports 80 and 443"
    },
    # 13. CI/CD → GitHub HTTPS
    {
        "action": "allow", "protocol": "tcp", "source": "cicd_server",
        "destination": "github.com", "source_port": "any",
        "destination_port": "443", "direction": "outbound",
        "description": "Allow CI/CD server to pull code from GitHub over HTTPS"
    },
    # 14. Deny DB server internet access — CORRECT
    {
        "action": "deny", "protocol": "any", "source": "db_server",
        "destination": "0.0.0.0/0", "source_port": "any",
        "destination_port": "any", "direction": "outbound",
        "description": "Deny direct internet access from database server"
    },
    # 15. Monitoring SNMP from all devices
    {
        "action": "allow", "protocol": "udp", "source": "network_devices",
        "destination": "monitoring_server", "source_port": "any",
        "destination_port": "161", "direction": "inbound",
        "description": "Allow SNMP data collection from network devices to monitoring server"
    },
    # 16. Block ICMP from external — HALLUCINATED: uses wrong protocol name
    {
        "action": "deny", "protocol": "tcp", "source": "any",
        "destination": "any", "source_port": "any",
        "destination_port": "any", "direction": "inbound",
        "description": "Block ping requests"
    },
    # 17. Security scanning all ports
    {
        "action": "allow", "protocol": "any", "source": "scanner_tool",
        "destination": "internal_servers", "source_port": "any",
        "destination_port": "any", "direction": "inbound",
        "description": "Allow security scanner to probe all ports on internal servers"
    },
    # 18. Block Telnet port 23
    {
        "action": "deny", "protocol": "tcp", "source": "any",
        "destination": "any", "source_port": "any",
        "destination_port": "23", "direction": "both",
        "description": "Block Telnet traffic across entire network"
    },
    # 19. Email server outbound SMTP 587
    {
        "action": "allow", "protocol": "tcp", "source": "mail_server",
        "destination": "any", "source_port": "any",
        "destination_port": "587", "direction": "outbound",
        "description": "Allow email server to send outbound SMTP traffic on port 587"
    },
    # 20. Block dev → prod — HALLUCINATED: allows instead of blocks
    {
        "action": "allow", "protocol": "any", "source": "dev_environment",
        "destination": "production_network", "source_port": "any",
        "destination_port": "any", "direction": "both",
        "description": "Allow traffic between dev and production for deployment"
    },
]

SEED_REQUIREMENTS = [
    "Allow employees to access the company website over HTTPS.",
    "Block all inbound SSH connections from the internet.",
    "Allow the database server to communicate with the application server on port 5432.",
    "Deny all outbound traffic to known malicious IP ranges.",
    "Allow DNS queries from internal network to the corporate DNS server.",
    "Allow VPN users to access internal HR systems.",
    "Block RDP access from any external IP address.",
    "Allow IT admins to SSH into servers only from the management VLAN.",
    "Allow remote workers to connect to the file server over port 445.",
    "Deny any traffic from guest WiFi to the internal corporate network.",
    "Allow web servers to send outbound HTTPS requests to payment APIs.",
    "Block HTTP traffic and allow only HTTPS for all web communication.",
    "Allow the CI/CD pipeline server to pull code from GitHub over HTTPS.",
    "Deny direct internet access from the database server.",
    "Allow monitoring server to collect SNMP data from all network devices.",
    "Block all ICMP ping requests from external sources.",
    "Allow security scanning tools to probe internal servers on all ports.",
    "Deny any traffic on port 23 (Telnet) across the entire network.",
    "Allow email server to send outbound SMTP traffic on port 587.",
    "Block all traffic between the development environment and the production network.",
]


def run_mock_pipeline(output_path: str = "../data/week1_seed_dataset.json"):
    """Generate mock Week 1 dataset without needing an API key."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = []
    for i, (req, rule) in enumerate(zip(SEED_REQUIREMENTS, MOCK_LLM_OUTPUTS), 1):
        pair = {
            "pair_id": f"W1-{i:03d}",
            "requirement": req,
            "generated_rule": rule,
            "raw_llm_output": json.dumps(rule),
            "generation_metadata": {
                "llm_backend": "mock",
                "model": "mock-v1",
                "timestamp": datetime.utcnow().isoformat(),
                "parse_success": True
            }
        }
        dataset.append(pair)
        print(f"[{i:02d}] {pair['pair_id']} ✓ {rule['action'].upper()} "
              f"{rule['protocol'].upper()} "
              f"{rule['source']} → {rule['destination']}:{rule['destination_port']}")

    output = {
        "metadata": {
            "created_at": datetime.utcnow().isoformat(),
            "total_pairs": len(dataset),
            "successful_parses": len(dataset),
            "llm_backend": "mock",
            "model": "mock-v1"
        },
        "pairs": dataset
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Mock dataset saved to: {output_path}")
    return dataset


if __name__ == "__main__":
    print("TrustGuard Week 1 - Mock Pipeline Runner")
    print("=" * 50)
    run_mock_pipeline()