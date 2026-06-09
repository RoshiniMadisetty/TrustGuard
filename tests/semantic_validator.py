"""
TrustGuard - Person 3, Week 2
Semantic Intent Matching Engine

Compares the generated firewall rule against the original
natural language requirement and flags mismatches.

Uses:
  - Keyword-based intent extraction (no model needed)
  - sentence-transformers for embedding similarity (optional, better accuracy)

Install (for embedding mode):
  pip install sentence-transformers

Run: python semantic_validator.py
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# INTENT MAPS
# ──────────────────────────────────────────────

ACTION_INTENT = {
    "block":    "deny",  "deny":     "deny",  "restrict": "deny",
    "prevent":  "deny",  "forbid":   "deny",  "disallow": "deny",
    "no access":"deny",  "isolate":  "deny",  "quarantine":"deny",
    "allow":    "allow", "permit":   "allow", "enable":   "allow",
    "grant":    "allow", "let":      "allow", "access":   "allow",
    "should access": "allow"
}

SERVICE_PORT_MAP = {
    "https":      ("tcp",  "443"),
    "http":       ("tcp",  "80"),
    "ssh":        ("tcp",  "22"),
    "rdp":        ("tcp",  "3389"),
    "dns":        ("udp",  "53"),
    "ftp":        ("tcp",  "21"),
    "sftp":       ("tcp",  "22"),
    "smtp":       ("tcp",  "25"),
    "port 587":   ("tcp",  "587"),
    "587":        ("tcp",  "587"),
    "telnet":     ("tcp",  "23"),
    "port 23":    ("tcp",  "23"),
    "snmp":       ("udp",  "161"),
    "smb":        ("tcp",  "445"),
    "port 445":   ("tcp",  "445"),
    "postgres":   ("tcp",  "5432"),
    "port 5432":  ("tcp",  "5432"),
    "mysql":      ("tcp",  "3306"),
    "port 3306":  ("tcp",  "3306"),
    "ldap":       ("tcp",  "389"),
    "ntp":        ("udp",  "123"),
    "syslog":     ("udp",  "514"),
    "ping":       ("icmp", "any"),
    "icmp":       ("icmp", "any"),
    "port 8080":  ("tcp",  "8080"),
    "port 8443":  ("tcp",  "8443"),
    "port 443":   ("tcp",  "443"),
    "port 80":    ("tcp",  "80"),
    "port 22":    ("tcp",  "22"),
}

SOURCE_CONSTRAINT_KEYWORDS = [
    "from the internet", "from external", "from outside",
    "only from", "from the management", "from vpn",
    "from admins", "from the", "guest wifi", "internal network",
    "employees", "remote workers", "finance", "accounting",
    "dev", "development", "partner"
]


# ──────────────────────────────────────────────
# MISMATCH RESULT
# ──────────────────────────────────────────────

@dataclass
class IntentMatchResult:
    matches: bool = True
    mismatches: list = field(default_factory=list)
    intent_extracted: dict = field(default_factory=dict)
    confidence: float = 1.0

    def add_mismatch(self, msg: str, severity: str = "high"):
        self.matches = False
        self.mismatches.append({"message": msg, "severity": severity})

    def to_dict(self) -> dict:
        return {
            "matches": self.matches,
            "mismatches": self.mismatches,
            "intent_extracted": self.intent_extracted,
            "confidence": self.confidence,
            "mismatch_count": len(self.mismatches)
        }


# ──────────────────────────────────────────────
# SEMANTIC INTENT MATCHER
# ──────────────────────────────────────────────

class SemanticIntentMatcher:
    """
    Compares natural language requirement against generated firewall rule.
    Flags semantic mismatches — wrong action, wrong port, wrong protocol,
    missing constraints, scope expansion.
    """

    def __init__(self, use_embeddings: bool = False):
        self.use_embeddings = use_embeddings
        self._model = None

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer, util
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._util = util
                print("  Embeddings: sentence-transformers loaded ✓")
            except ImportError:
                print("  Warning: sentence-transformers not installed. "
                      "Falling back to keyword matching.")
                self.use_embeddings = False

    def match(self, requirement: str, rule: Optional[dict]) -> IntentMatchResult:
        result = IntentMatchResult()

        if rule is None:
            result.add_mismatch("No rule generated — cannot match intent", "critical")
            result.confidence = 0.0
            return result

        req_lower = requirement.lower()

        # ── Extract intent from requirement ───────────────────────────────────
        expected_action   = self._extract_action(req_lower)
        expected_protocol, expected_port = self._extract_service(req_lower)
        has_source_constraint = self._has_source_constraint(req_lower)

        result.intent_extracted = {
            "expected_action":   expected_action,
            "expected_protocol": expected_protocol,
            "expected_port":     expected_port,
            "has_source_constraint": has_source_constraint
        }

        # ── Check 1: Action intent flip ───────────────────────────────────────
        rule_action = str(rule.get("action", "")).lower()
        if expected_action:
            expected_deny  = expected_action == "deny"
            rule_allows    = rule_action == "allow"
            rule_denies    = rule_action in {"deny", "drop", "reject"}

            if expected_deny and rule_allows:
                result.add_mismatch(
                    f"INTENT FLIP: Requirement says '{self._get_action_keyword(req_lower)}' "
                    f"but rule action is '{rule_action}'",
                    "critical"
                )
            elif not expected_deny and rule_denies:
                result.add_mismatch(
                    f"INTENT FLIP: Requirement says allow/permit "
                    f"but rule action is '{rule_action}'",
                    "critical"
                )

        # ── Check 2: Protocol mismatch ────────────────────────────────────────
        rule_protocol = str(rule.get("protocol", "")).lower()
        if expected_protocol and rule_protocol != "any":
            if rule_protocol != expected_protocol:
                result.add_mismatch(
                    f"WRONG PROTOCOL: Requirement implies '{expected_protocol}' "
                    f"but rule uses '{rule_protocol}'",
                    "high"
                )

        # ── Check 3: Port mismatch ────────────────────────────────────────────
        rule_port = str(rule.get("destination_port", "any")).lower()
        if expected_port and expected_port != "any":
            if rule_port not in {"any", "*"} and expected_port not in rule_port:
                result.add_mismatch(
                    f"WRONG PORT: Requirement implies port {expected_port} "
                    f"but rule uses port {rule_port}",
                    "high"
                )

        # ── Check 4: Missing source constraint ────────────────────────────────
        rule_source = str(rule.get("source", "any")).lower()
        if has_source_constraint and rule_source in {"any", "*", "0.0.0.0/0", "all"}:
            result.add_mismatch(
                "MISSING CONSTRAINT: Requirement specifies a source restriction "
                "but rule uses 'any' for source",
                "high"
            )

        # ── Check 5: Scope expansion ──────────────────────────────────────────
        rule_dest = str(rule.get("destination", "any")).lower()
        if self._has_specific_destination(req_lower):
            if rule_dest in {"any", "*", "0.0.0.0/0", "all"}:
                result.add_mismatch(
                    "SCOPE EXPANSION: Requirement mentions a specific destination "
                    "but rule destination is 'any'",
                    "medium"
                )

        # ── Check 6: HTTPS-only downgrade ─────────────────────────────────────
        if ("only https" in req_lower or "https only" in req_lower
                or ("block http" in req_lower and "https" in req_lower)):
            if "80" in rule_port or rule_protocol == "http":
                result.add_mismatch(
                    "SECURITY DOWNGRADE: Requirement asked for HTTPS-only "
                    "but rule allows port 80 (HTTP)",
                    "high"
                )

        # ── Embedding similarity (optional) ──────────────────────────────────
        if self.use_embeddings and self._model:
            sim_score = self._embedding_similarity(requirement, rule)
            result.confidence = sim_score
            if sim_score < 0.4:
                result.add_mismatch(
                    f"LOW SEMANTIC SIMILARITY: Embedding score {sim_score:.2f} < 0.4 "
                    f"— rule may not reflect requirement intent",
                    "medium"
                )
        else:
            # Keyword-based confidence: penalise per mismatch
            result.confidence = max(0.0, 1.0 - len(result.mismatches) * 0.25)

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_action(self, req: str) -> Optional[str]:
        for keyword, action in ACTION_INTENT.items():
            if keyword in req:
                return action
        return None

    def _get_action_keyword(self, req: str) -> str:
        for keyword in ACTION_INTENT:
            if keyword in req:
                return keyword
        return "unknown"

    def _extract_service(self, req: str) -> tuple:
        for service, (proto, port) in SERVICE_PORT_MAP.items():
            if service in req:
                return proto, port
        # Check for bare port numbers
        port_match = re.search(r"port\s+(\d+)", req)
        if port_match:
            return "tcp", port_match.group(1)
        return None, None

    def _has_source_constraint(self, req: str) -> bool:
        return any(kw in req for kw in SOURCE_CONSTRAINT_KEYWORDS)

    def _has_specific_destination(self, req: str) -> bool:
        specific = [
            "web server", "database", "hr system", "file server", "dns server",
            "monitoring server", "mail server", "github", "payment", "backup server",
            "ldap server", "time server", "siem", "cardholder"
        ]
        return any(t in req for t in specific)

    def _embedding_similarity(self, requirement: str, rule: dict) -> float:
        """Compute cosine similarity between requirement and rule description."""
        rule_text = (
            f"{rule.get('action','')} {rule.get('protocol','')} "
            f"from {rule.get('source','')} to {rule.get('destination','')} "
            f"port {rule.get('destination_port','')}. "
            f"{rule.get('description','')}"
        )
        emb1 = self._model.encode(requirement, convert_to_tensor=True)
        emb2 = self._model.encode(rule_text, convert_to_tensor=True)
        score = float(self._util.cos_sim(emb1, emb2))
        return round(score, 4)


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_semantic_validation(
    dataset_path: str = "../person1_llm_pipeline/data/week2_labeled_dataset.json",
    output_path: str = "../outputs/semantic_validation_results.json",
    use_embeddings: bool = False
):
    import os
    from datetime import datetime, UTC

    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    matcher = SemanticIntentMatcher(use_embeddings=use_embeddings)

    results = []
    stats = {"match": 0, "mismatch": 0}
    correct_predictions = 0   # How often our mismatch = person1's label

    print(f"\n{'='*65}")
    print(f"TrustGuard Person 3 — Semantic Intent Matcher")
    print(f"Mode: {'Embeddings' if use_embeddings else 'Keyword matching'}")
    print(f"Validating {len(pairs)} pairs...")
    print(f"{'='*65}\n")

    for pair in pairs:
        req  = pair["requirement"]
        rule = pair.get("generated_rule")
        p1_label = pair.get("label", "unknown")

        imr = matcher.match(req, rule)

        # Compare our prediction vs Person 1's label
        our_prediction = "hallucinated" if not imr.matches else "correct"
        # Person 1 labels dangerous + hallucinated = not correct
        p1_not_correct = p1_label in {"hallucinated", "dangerous"}
        agreement = (our_prediction == "correct") == (not p1_not_correct)
        if agreement:
            correct_predictions += 1

        if imr.matches:
            stats["match"] += 1
            flag = "✓"
        else:
            stats["mismatch"] += 1
            flag = "✗"

        print(f"  [{flag}] {pair['pair_id']} | P1:{p1_label:<12} | "
              f"Conf:{imr.confidence:.2f} | {req[:45]}...")
        for mm in imr.mismatches:
            print(f"        → {mm['severity'].upper()}: {mm['message']}")

        results.append({
            "pair_id": pair["pair_id"],
            "requirement": req,
            "p1_label": p1_label,
            "p1_hallucination_type": pair.get("hallucination_type", "none"),
            "semantic_match": imr.matches,
            "confidence": imr.confidence,
            "intent_extracted": imr.intent_extracted,
            "mismatches": imr.mismatches,
            "our_prediction": our_prediction,
            "agrees_with_p1": agreement
        })

    accuracy = correct_predictions / len(pairs) * 100
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "total": len(results),
            "semantic_matches": stats["match"],
            "semantic_mismatches": stats["mismatch"],
            "agreement_with_p1_labels": f"{accuracy:.1f}%",
            "embedding_mode": use_embeddings
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✓ Semantic matches:    {stats['match']}")
    print(f"  ✗ Semantic mismatches: {stats['mismatch']}")
    print(f"  Agreement with P1 labels: {accuracy:.1f}%")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*65}\n")

    return output


if __name__ == "__main__":
    run_semantic_validation(use_embeddings=False)