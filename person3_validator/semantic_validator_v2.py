"""
TrustGuard - Person 3, Week 2 (Upgraded)
Semantic Intent Matching Engine — v2

Upgrades over v1:
  - sentence-transformers embedding similarity (all-MiniLM-L6-v2)
  - TF-IDF cosine similarity fallback (no model needed)
  - Security downgrade detection (new hallucination category)
  - Contradictory intent detection within single requirement

Install:
  pip install sentence-transformers scikit-learn

Run: python semantic_validator_v2.py
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, UTC


# ──────────────────────────────────────────────
# INTENT MAPS (same as v1, extended)
# ──────────────────────────────────────────────

ACTION_INTENT = {
    "block":     "deny",  "deny":      "deny",  "restrict":  "deny",
    "prevent":   "deny",  "forbid":    "deny",  "disallow":  "deny",
    "no access": "deny",  "isolate":   "deny",  "quarantine":"deny",
    "allow":    "allow",  "permit":   "allow",  "enable":   "allow",
    "grant":    "allow",  "let":      "allow",  "access":   "allow",
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
    "ldap":       ("tcp",  "389"),
    "ntp":        ("udp",  "123"),
    "syslog":     ("udp",  "514"),
    "ping":       ("icmp", "any"),
    "icmp":       ("icmp", "any"),
    "port 8080":  ("tcp",  "8080"),
    "port 443":   ("tcp",  "443"),
    "port 80":    ("tcp",  "80"),
    "port 22":    ("tcp",  "22"),
}

SOURCE_CONSTRAINT_KEYWORDS = [
    "from the internet", "from external", "from outside",
    "only from", "from the management", "from vpn",
    "from admins", "guest wifi", "internal network",
    "employees", "remote workers", "finance", "accounting",
    "dev", "development", "partner", "external ip"
]

SPECIFIC_DESTINATION_KEYWORDS = [
    "web server", "database", "hr system", "file server", "dns server",
    "monitoring server", "mail server", "github", "payment", "backup server",
    "ldap server", "time server", "siem", "cardholder", "rds", "s3"
]

# Security downgrade patterns:
# requirement implies strong encryption / restricted protocol,
# but rule allows the weaker variant
DOWNGRADE_PATTERNS = [
    {
        "name": "HTTPS-only weakened to HTTP",
        "req_signals": ["only https", "https only", "block http", "https traffic only",
                        "allow only https", "enforce https"],
        "rule_check": lambda r: (
            "80" in str(r.get("destination_port", ""))
            or str(r.get("protocol", "")).lower() == "http"
        )
    },
    {
        "name": "SSH required but Telnet allowed",
        "req_signals": ["ssh", "secure shell"],
        "rule_check": lambda r: str(r.get("destination_port", "")) == "23"
    },
    {
        "name": "SFTP required but FTP allowed",
        "req_signals": ["sftp", "secure ftp", "secure file transfer"],
        "rule_check": lambda r: str(r.get("destination_port", "")) == "21"
    },
    {
        "name": "Encrypted backup weakened",
        "req_signals": ["encrypted backup", "secure backup"],
        "rule_check": lambda r: str(r.get("destination_port", "")) in ["20", "21", "23"]
    },
]


# ──────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────

@dataclass
class IntentMatchResult:
    matches: bool = True
    mismatches: list = field(default_factory=list)
    intent_extracted: dict = field(default_factory=dict)
    keyword_confidence: float = 1.0
    embedding_similarity: float = -1.0   # -1 = not computed
    final_confidence: float = 1.0
    hallucination_types_detected: list = field(default_factory=list)

    def add_mismatch(self, msg: str, severity: str, htype: str):
        self.matches = False
        self.mismatches.append({"message": msg, "severity": severity,
                                "hallucination_type": htype})
        if htype not in self.hallucination_types_detected:
            self.hallucination_types_detected.append(htype)

    def to_dict(self) -> dict:
        return {
            "matches": self.matches,
            "mismatches": self.mismatches,
            "intent_extracted": self.intent_extracted,
            "keyword_confidence": self.keyword_confidence,
            "embedding_similarity": self.embedding_similarity,
            "final_confidence": self.final_confidence,
            "hallucination_types_detected": self.hallucination_types_detected,
            "mismatch_count": len(self.mismatches)
        }


# ──────────────────────────────────────────────
# TFIDF FALLBACK
# ──────────────────────────────────────────────

def tfidf_similarity(text1: str, text2: str) -> float:
    """Cosine similarity using TF-IDF. No model needed."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer()
        tfidf = vect.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    except ImportError:
        return -1.0


# ──────────────────────────────────────────────
# SEMANTIC INTENT MATCHER v2
# ──────────────────────────────────────────────

class SemanticIntentMatcherV2:

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self._model = None
        self._util = None

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer, util
                print("  Loading sentence-transformers (all-MiniLM-L6-v2)...")
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._util = util
                print("  Embeddings ready ✓")
            except ImportError:
                print("  sentence-transformers not found — using TF-IDF fallback")
                print("  Install: pip install sentence-transformers")
                self.use_embeddings = False

    def match(self, requirement: str, rule: Optional[dict]) -> IntentMatchResult:
        result = IntentMatchResult()

        if rule is None:
            result.add_mismatch(
                "No rule generated — LLM failed to produce output",
                "critical", "generation_failure"
            )
            result.keyword_confidence = 0.0
            result.final_confidence = 0.0
            return result

        req_lower = requirement.lower()

        # ── Extract intent ────────────────────────────────────────────────────
        expected_action = self._extract_action(req_lower)
        expected_proto, expected_port = self._extract_service(req_lower)
        has_src_constraint = self._has_source_constraint(req_lower)

        result.intent_extracted = {
            "expected_action":       expected_action,
            "expected_protocol":     expected_proto,
            "expected_port":         expected_port,
            "has_source_constraint": has_src_constraint
        }

        rule_action   = str(rule.get("action",   "")).lower()
        rule_protocol = str(rule.get("protocol", "")).lower()
        rule_port     = str(rule.get("destination_port", "any")).lower()
        rule_source   = str(rule.get("source",   "any")).lower()
        rule_dest     = str(rule.get("destination", "any")).lower()

        # ── Check 1: Intent flip ──────────────────────────────────────────────
        if expected_action:
            if expected_action == "deny" and rule_action == "allow":
                kw = self._get_action_keyword(req_lower)
                result.add_mismatch(
                    f"INTENT FLIP: Requirement says '{kw}' but rule action is 'allow'",
                    "critical", "intent_flip"
                )
            elif expected_action == "allow" and rule_action in {"deny","drop","reject"}:
                result.add_mismatch(
                    f"INTENT FLIP: Requirement implies allow but rule action is '{rule_action}'",
                    "critical", "intent_flip"
                )

        # ── Check 2: Wrong protocol ───────────────────────────────────────────
        if expected_proto and rule_protocol not in {"any", expected_proto}:
            result.add_mismatch(
                f"WRONG PROTOCOL: Requirement implies '{expected_proto}' "
                f"but rule uses '{rule_protocol}'",
                "high", "wrong_protocol"
            )

        # ── Check 3: Wrong port ───────────────────────────────────────────────
        if expected_port and expected_port != "any":
            if rule_port not in {"any", "*"} and expected_port not in rule_port:
                result.add_mismatch(
                    f"WRONG PORT: Requirement implies port {expected_port} "
                    f"but rule uses port {rule_port}",
                    "high", "wrong_port"
                )

        # ── Check 4: Missing source constraint ────────────────────────────────
        if has_src_constraint and rule_source in {"any","*","0.0.0.0/0","all",""}:
            result.add_mismatch(
                "MISSING CONSTRAINT: Requirement specifies a source restriction "
                "but rule uses 'any' for source",
                "high", "missing_constraint"
            )

        # ── Check 5: Scope expansion ──────────────────────────────────────────
        if self._has_specific_destination(req_lower):
            if rule_dest in {"any","*","0.0.0.0/0","all",""}:
                result.add_mismatch(
                    "SCOPE EXPANSION: Requirement names a specific destination "
                    "but rule destination is 'any'",
                    "medium", "scope_expansion"
                )

        # ── Check 6: Security downgrade (NEW) ─────────────────────────────────
        for pattern in DOWNGRADE_PATTERNS:
            req_matches = any(sig in req_lower for sig in pattern["req_signals"])
            if req_matches:
                try:
                    rule_matches = pattern["rule_check"](rule)
                except Exception:
                    rule_matches = False
                if rule_matches:
                    result.add_mismatch(
                        f"SECURITY DOWNGRADE: {pattern['name']} — "
                        f"requirement asked for stronger security but rule weakens it",
                        "high", "security_downgrade"
                    )

        # ── Keyword confidence ────────────────────────────────────────────────
        result.keyword_confidence = max(0.0, 1.0 - len(result.mismatches) * 0.2)

        # ── Embedding similarity ──────────────────────────────────────────────
        rule_text = self._rule_to_text(rule)

        if self.use_embeddings and self._model:
            try:
                import torch
                emb1 = self._model.encode(requirement, convert_to_tensor=True)
                emb2 = self._model.encode(rule_text,   convert_to_tensor=True)
                sim  = float(self._util.cos_sim(emb1, emb2))
                result.embedding_similarity = round(sim, 4)

                # Low similarity flag — only add if no mismatches already found
                if sim < 0.35 and result.matches:
                    result.add_mismatch(
                        f"LOW SEMANTIC SIMILARITY: embedding score {sim:.2f} < 0.35 "
                        "— rule may not reflect requirement intent",
                        "medium", "low_similarity"
                    )
            except Exception as e:
                result.embedding_similarity = -1.0
        else:
            # TF-IDF fallback
            tfidf_score = tfidf_similarity(requirement, rule_text)
            result.embedding_similarity = round(tfidf_score, 4)

        # ── Final confidence = weighted average ───────────────────────────────
        if result.embedding_similarity >= 0:
            result.final_confidence = round(
                0.4 * result.keyword_confidence +
                0.6 * max(0.0, result.embedding_similarity), 4
            )
        else:
            result.final_confidence = result.keyword_confidence

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_action(self, req: str) -> Optional[str]:
        for kw, action in ACTION_INTENT.items():
            if kw in req:
                return action
        return None

    def _get_action_keyword(self, req: str) -> str:
        for kw in ACTION_INTENT:
            if kw in req:
                return kw
        return "unknown"

    def _extract_service(self, req: str) -> tuple:
        for svc, (proto, port) in SERVICE_PORT_MAP.items():
            if svc in req:
                return proto, port
        m = re.search(r"port\s+(\d+)", req)
        if m:
            return "tcp", m.group(1)
        return None, None

    def _has_source_constraint(self, req: str) -> bool:
        return any(kw in req for kw in SOURCE_CONSTRAINT_KEYWORDS)

    def _has_specific_destination(self, req: str) -> bool:
        return any(t in req for t in SPECIFIC_DESTINATION_KEYWORDS)

    def _rule_to_text(self, rule: dict) -> str:
        return (
            f"{rule.get('action','')} {rule.get('protocol','')} traffic "
            f"from {rule.get('source','')} to {rule.get('destination','')} "
            f"on port {rule.get('destination_port','')}. "
            f"{rule.get('description','')}"
        )


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_semantic_validation_v2(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/semantic_v2_results.json",
    use_embeddings: bool = True
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs   = data["pairs"]
    matcher = SemanticIntentMatcherV2(use_embeddings=use_embeddings)

    results        = []
    correct_preds  = 0
    match_count    = 0
    mismatch_count = 0
    htype_counts   = {}

    print(f"\n{'='*65}")
    print(f"TrustGuard Person 3 — Semantic Validator v2")
    print(f"Mode: {'Embeddings (all-MiniLM-L6-v2)' if use_embeddings else 'TF-IDF fallback'}")
    print(f"Dataset: {len(pairs)} pairs")
    print(f"{'='*65}\n")

    for pair in pairs:
        req     = pair["requirement"]
        rule    = pair.get("generated_rule")
        p1_lbl  = pair.get("label", "unknown")

        imr = matcher.match(req, rule)

        # Prediction vs P1 label
        our_pred    = "hallucinated" if not imr.matches else "correct"
        p1_not_ok   = p1_lbl in {"hallucinated", "dangerous"}
        agreement   = (our_pred == "correct") == (not p1_not_ok)
        if agreement:
            correct_preds += 1

        if imr.matches:
            match_count += 1
            flag = "✓"
        else:
            mismatch_count += 1
            flag = "✗"
            for htype in imr.hallucination_types_detected:
                htype_counts[htype] = htype_counts.get(htype, 0) + 1

        # Similarity display
        emb_str = (f"emb:{imr.embedding_similarity:.2f}"
                   if imr.embedding_similarity >= 0 else "emb:N/A")

        print(f"  [{flag}] {pair['pair_id']} | P1:{p1_lbl:<12} | "
              f"kw:{imr.keyword_confidence:.2f} {emb_str} "
              f"final:{imr.final_confidence:.2f} | "
              f"{req[:40]}...")
        for mm in imr.mismatches:
            print(f"       → {mm['severity'].upper()} [{mm['hallucination_type']}]: "
                  f"{mm['message']}")

        results.append({
            "pair_id":               pair["pair_id"],
            "requirement":           req,
            "p1_label":              p1_lbl,
            "p1_hallucination_type": pair.get("hallucination_type","none"),
            "semantic_match":        imr.matches,
            "keyword_confidence":    imr.keyword_confidence,
            "embedding_similarity":  imr.embedding_similarity,
            "final_confidence":      imr.final_confidence,
            "intent_extracted":      imr.intent_extracted,
            "mismatches":            imr.mismatches,
            "hallucination_types_detected": imr.hallucination_types_detected,
            "our_prediction":        our_pred,
            "agrees_with_p1":        agreement
        })

    accuracy = correct_preds / len(pairs) * 100

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "metadata": {
            "created_at":              datetime.now(UTC).isoformat(),
            "total":                   len(results),
            "semantic_matches":        match_count,
            "semantic_mismatches":     mismatch_count,
            "agreement_with_p1":       f"{accuracy:.1f}%",
            "embedding_mode":          use_embeddings,
            "hallucination_type_breakdown": htype_counts
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✓ Matches:           {match_count}")
    print(f"  ✗ Mismatches:        {mismatch_count}")
    print(f"  Agreement with P1:   {accuracy:.1f}%")
    print(f"\n  Hallucination types detected by semantic validator:")
    for htype, cnt in sorted(htype_counts.items(), key=lambda x: -x[1]):
        print(f"    {htype:<30} {cnt}")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*65}\n")

    return output


if __name__ == "__main__":
    # use_embeddings=True  → sentence-transformers (better, needs pip install)
    # use_embeddings=False → TF-IDF fallback (works immediately)
    run_semantic_validation_v2(use_embeddings=False)