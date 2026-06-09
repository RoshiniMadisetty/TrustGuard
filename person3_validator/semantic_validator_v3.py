"""
TrustGuard - Person 3
Semantic Validator v3

Upgrades over v2:
  - sentence-transformers as PRIMARY (not fallback)
  - TF-IDF as fallback only if transformers unavailable
  - security_downgrade as explicit hallucination category
  - Confidence threshold tuned from real dataset results
  - Weighted final score: 60% embedding + 40% keyword

Install:
  pip install sentence-transformers scikit-learn

Run: python semantic_validator_v3.py
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, UTC


# ──────────────────────────────────────────────
# INTENT MAPS
# ──────────────────────────────────────────────

ACTION_INTENT = {
    "block":         "deny",   "deny":        "deny",
    "restrict":      "deny",   "prevent":     "deny",
    "forbid":        "deny",   "disallow":    "deny",
    "no access":     "deny",   "isolate":     "deny",
    "quarantine":    "deny",   "stop":        "deny",
    "drop":          "deny",   "reject":      "deny",
    "allow":         "allow",  "permit":      "allow",
    "enable":        "allow",  "grant":       "allow",
    "let":           "allow",  "access":      "allow",
    "should access": "allow",  "open":        "allow",
}

SERVICE_PORT_MAP = {
    "https":        ("tcp",  "443"),
    "http":         ("tcp",  "80"),
    "ssh":          ("tcp",  "22"),
    "secure shell": ("tcp",  "22"),
    "rdp":          ("tcp",  "3389"),
    "remote desktop": ("tcp","3389"),
    "dns":          ("udp",  "53"),
    "ftp":          ("tcp",  "21"),
    "sftp":         ("tcp",  "22"),
    "smtp":         ("tcp",  "25"),
    "port 587":     ("tcp",  "587"),
    "587":          ("tcp",  "587"),
    "telnet":       ("tcp",  "23"),
    "port 23":      ("tcp",  "23"),
    "snmp":         ("udp",  "161"),
    "smb":          ("tcp",  "445"),
    "port 445":     ("tcp",  "445"),
    "postgres":     ("tcp",  "5432"),
    "port 5432":    ("tcp",  "5432"),
    "mysql":        ("tcp",  "3306"),
    "ldap":         ("tcp",  "389"),
    "ntp":          ("udp",  "123"),
    "syslog":       ("udp",  "514"),
    "ping":         ("icmp", "any"),
    "icmp":         ("icmp", "any"),
    "port 8080":    ("tcp",  "8080"),
    "port 443":     ("tcp",  "443"),
    "port 80":      ("tcp",  "80"),
    "port 22":      ("tcp",  "22"),
}

SOURCE_CONSTRAINT_KEYWORDS = [
    "from the internet", "from external", "from outside",
    "only from", "from the management", "from vpn",
    "from admins", "guest wifi", "internal network",
    "employees", "remote workers", "finance", "accounting",
    "dev", "development", "partner", "external ip",
    "from the internet", "external sources",
]

SPECIFIC_DESTINATION_KEYWORDS = [
    "web server", "database", "hr system", "file server", "dns server",
    "monitoring server", "mail server", "github", "payment", "backup server",
    "ldap server", "time server", "siem", "cardholder", "rds", "s3",
]

# ── Security downgrade patterns ────────────────────────────────────────────────
# Each entry: requirement signals + rule check function
DOWNGRADE_PATTERNS = [
    {
        "name":        "HTTPS-only weakened to HTTP",
        "req_signals": [
            "only https", "https only", "block http", "https traffic only",
            "allow only https", "enforce https", "https and not http"
        ],
        "rule_check": lambda r: (
            "80" in str(r.get("destination_port", ""))
            or str(r.get("protocol", "")).lower() == "http"
        ),
        "severity": "high"
    },
    {
        "name":        "SSH required but Telnet allowed",
        "req_signals": ["ssh", "secure shell", "remote administration"],
        "rule_check":  lambda r: str(r.get("destination_port", "")) == "23",
        "severity":    "critical"
    },
    {
        "name":        "SFTP required but FTP allowed",
        "req_signals": ["sftp", "secure ftp", "secure file transfer"],
        "rule_check":  lambda r: str(r.get("destination_port", "")) == "21",
        "severity":    "critical"
    },
    {
        "name":        "Encrypted comms weakened to plaintext port",
        "req_signals": ["encrypted", "secure", "tls", "ssl"],
        "rule_check":  lambda r: str(r.get("destination_port", "")) in {"23","21","80"},
        "severity":    "high"
    },
]


# ──────────────────────────────────────────────
# RESULT
# ──────────────────────────────────────────────

@dataclass
class IntentMatchResult:
    matches:                    bool  = True
    mismatches:                 list  = field(default_factory=list)
    intent_extracted:           dict  = field(default_factory=dict)
    keyword_confidence:         float = 1.0
    embedding_similarity:       float = -1.0
    final_confidence:           float = 1.0
    hallucination_types_detected: list = field(default_factory=list)
    embedding_backend:          str   = "none"

    def add_mismatch(self, msg: str, severity: str, htype: str):
        self.matches = False
        self.mismatches.append({
            "message":          msg,
            "severity":         severity,
            "hallucination_type": htype
        })
        if htype not in self.hallucination_types_detected:
            self.hallucination_types_detected.append(htype)

    def to_dict(self) -> dict:
        return {
            "matches":                      self.matches,
            "mismatches":                   self.mismatches,
            "intent_extracted":             self.intent_extracted,
            "keyword_confidence":           self.keyword_confidence,
            "embedding_similarity":         self.embedding_similarity,
            "final_confidence":             self.final_confidence,
            "hallucination_types_detected": self.hallucination_types_detected,
            "embedding_backend":            self.embedding_backend,
            "mismatch_count":               len(self.mismatches)
        }


# ──────────────────────────────────────────────
# EMBEDDING BACKENDS
# ──────────────────────────────────────────────

def _load_sentence_transformer():
    """Try to load sentence-transformers. Returns (model, util) or (None, None)."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, util
    except Exception:
        return None, None


def _tfidf_similarity(t1: str, t2: str) -> float:
    """TF-IDF cosine similarity fallback."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        v = TfidfVectorizer()
        m = v.fit_transform([t1, t2])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except Exception:
        return -1.0


# ──────────────────────────────────────────────
# SEMANTIC VALIDATOR v3
# ──────────────────────────────────────────────

class SemanticValidatorV3:

    # Similarity threshold — below this, flag as low similarity even if
    # no keyword mismatch found (catches cases keyword matching misses)
    EMBEDDING_LOW_THRESHOLD = 0.38

    def __init__(self):
        print("  Loading sentence-transformers (all-MiniLM-L6-v2)...")
        self._st_model, self._st_util = _load_sentence_transformer()
        if self._st_model:
            print("  ✓ sentence-transformers ready — embedding mode active")
            self._backend = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            print("  ⚠ sentence-transformers unavailable — using TF-IDF fallback")
            print("    Install: pip install sentence-transformers")
            self._backend = "tfidf-fallback"

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def match(self, requirement: str, rule: Optional[dict]) -> IntentMatchResult:
        result = IntentMatchResult(embedding_backend=self._backend)

        if rule is None:
            result.add_mismatch(
                "No rule generated — LLM failed to produce output",
                "critical", "generation_failure"
            )
            result.keyword_confidence   = 0.0
            result.embedding_similarity = 0.0
            result.final_confidence     = 0.0
            return result

        req_low  = requirement.lower()
        action   = str(rule.get("action",              "")).lower()
        protocol = str(rule.get("protocol",            "")).lower()
        port     = str(rule.get("destination_port", "any")).lower()
        source   = str(rule.get("source",          "any")).lower()
        dest     = str(rule.get("destination",     "any")).lower()
        wildcard = {"any", "*", "0.0.0.0/0", "all", ""}

        # ── Extract intent ────────────────────────────────────────────────────
        exp_action         = self._extract_action(req_low)
        exp_proto, exp_port = self._extract_service(req_low)
        has_src_constraint = self._has_source_constraint(req_low)

        result.intent_extracted = {
            "expected_action":       exp_action,
            "expected_protocol":     exp_proto,
            "expected_port":         exp_port,
            "has_source_constraint": has_src_constraint,
        }

        # ── Keyword checks ────────────────────────────────────────────────────

        # 1. Intent flip
        if exp_action == "deny" and action == "allow":
            result.add_mismatch(
                f"INTENT FLIP: requirement says "
                f"'{self._get_action_kw(req_low)}' but rule action is 'allow'",
                "critical", "intent_flip"
            )
        elif exp_action == "allow" and action in {"deny","drop","reject"}:
            result.add_mismatch(
                f"INTENT FLIP: requirement implies allow but rule action is '{action}'",
                "critical", "intent_flip"
            )

        # 2. Wrong protocol
        if exp_proto and protocol not in {"any", exp_proto}:
            result.add_mismatch(
                f"WRONG PROTOCOL: requirement implies '{exp_proto}' "
                f"but rule uses '{protocol}'",
                "high", "wrong_protocol"
            )

        # 3. Wrong port
        if exp_port and exp_port != "any":
            if port not in {"any","*"} and exp_port not in port:
                result.add_mismatch(
                    f"WRONG PORT: requirement implies port {exp_port} "
                    f"but rule uses port {port}",
                    "high", "wrong_port"
                )

        # 4. Missing source constraint
        if has_src_constraint and source in wildcard:
            result.add_mismatch(
                "MISSING CONSTRAINT: requirement specifies a source restriction "
                "but rule uses 'any' for source",
                "high", "missing_constraint"
            )

        # 5. Scope expansion
        if self._has_specific_dest(req_low) and dest in wildcard:
            result.add_mismatch(
                "SCOPE EXPANSION: requirement names a specific destination "
                "but rule destination is 'any'",
                "medium", "scope_expansion"
            )

        # 6. Security downgrade (explicit category — not just scope expansion)
        for dp in DOWNGRADE_PATTERNS:
            if any(sig in req_low for sig in dp["req_signals"]):
                try:
                    triggered = dp["rule_check"](rule)
                except Exception:
                    triggered = False
                if triggered:
                    result.add_mismatch(
                        f"SECURITY DOWNGRADE — {dp['name']}: "
                        "requirement enforces stronger security but rule weakens it",
                        dp["severity"], "security_downgrade"
                    )
                    break   # one downgrade flag per rule is enough

        # ── Keyword confidence ────────────────────────────────────────────────
        result.keyword_confidence = round(
            max(0.0, 1.0 - len(result.mismatches) * 0.2), 4
        )

        # ── Embedding similarity ──────────────────────────────────────────────
        rule_text = self._rule_to_text(rule)
        sim = self._compute_similarity(requirement, rule_text)
        result.embedding_similarity = round(sim, 4)

        # Low similarity catch — only if keyword checks didn't already flag
        if sim >= 0 and sim < self.EMBEDDING_LOW_THRESHOLD and result.matches:
            result.add_mismatch(
                f"LOW SEMANTIC SIMILARITY: embedding score {sim:.3f} < "
                f"{self.EMBEDDING_LOW_THRESHOLD} — rule likely doesn't reflect "
                "requirement intent (caught by embeddings, missed by keywords)",
                "medium", "low_similarity"
            )

        # ── Final confidence: 60% embedding + 40% keyword ────────────────────
        if sim >= 0:
            result.final_confidence = round(
                0.6 * max(0.0, sim) + 0.4 * result.keyword_confidence, 4
            )
        else:
            result.final_confidence = result.keyword_confidence

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_similarity(self, t1: str, t2: str) -> float:
        if self._st_model:
            try:
                e1 = self._st_model.encode(t1, convert_to_tensor=True)
                e2 = self._st_model.encode(t2, convert_to_tensor=True)
                return float(self._st_util.cos_sim(e1, e2))
            except Exception:
                pass
        return _tfidf_similarity(t1, t2)

    def _rule_to_text(self, rule: dict) -> str:
        return (
            f"{rule.get('action','')} {rule.get('protocol','')} traffic "
            f"from {rule.get('source','')} "
            f"to {rule.get('destination','')} "
            f"port {rule.get('destination_port','')}. "
            f"{rule.get('description','')}"
        )

    def _extract_action(self, req: str) -> Optional[str]:
        for kw, act in ACTION_INTENT.items():
            if kw in req:
                return act
        return None

    def _get_action_kw(self, req: str) -> str:
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

    def _has_specific_dest(self, req: str) -> bool:
        return any(t in req for t in SPECIFIC_DESTINATION_KEYWORDS)


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────

def run_semantic_v3(
    dataset_path: str = "../person1_llm_pipeline/data/week4_final_dataset.json",
    output_path:  str = "../outputs/semantic_v3_results.json"
):
    with open(dataset_path) as f:
        data = json.load(f)

    pairs     = data["pairs"]
    validator = SemanticValidatorV3()

    results        = []
    match_ct       = 0
    mismatch_ct    = 0
    correct_preds  = 0
    htype_counts   = {}

    # Track per-category agreement for paper metrics
    category_stats = {}

    print(f"\n{'='*68}")
    print(f"TrustGuard — Semantic Validator v3")
    print(f"Backend: {validator._backend}")
    print(f"Dataset: {len(pairs)} pairs")
    print(f"{'='*68}\n")

    for pair in pairs:
        req    = pair["requirement"]
        rule   = pair.get("generated_rule")
        p1_lbl = pair.get("label", "unknown")
        p1_ht  = pair.get("hallucination_type", "none")

        imr = validator.match(req, rule)

        # Prediction vs P1 label
        our_pred  = "hallucinated" if not imr.matches else "correct"
        p1_not_ok = p1_lbl in {"hallucinated", "dangerous"}
        agreement = (our_pred == "correct") == (not p1_not_ok)
        if agreement:
            correct_preds += 1

        # Per-category agreement tracking
        cat = p1_ht if p1_ht != "none" else "correct"
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "agreed": 0}
        category_stats[cat]["total"]  += 1
        category_stats[cat]["agreed"] += int(agreement)

        if imr.matches:
            match_ct += 1
            flag = "✓"
        else:
            mismatch_ct += 1
            flag = "✗"
            for ht in imr.hallucination_types_detected:
                htype_counts[ht] = htype_counts.get(ht, 0) + 1

        emb_str = (f"emb:{imr.embedding_similarity:.3f}"
                   if imr.embedding_similarity >= 0 else "emb:N/A")
        print(f"  [{flag}] {pair['pair_id']} | P1:{p1_lbl:<12} | "
              f"kw:{imr.keyword_confidence:.2f} {emb_str} "
              f"→ final:{imr.final_confidence:.3f} | "
              f"{req[:38]}...")
        for mm in imr.mismatches:
            print(f"       → {mm['severity'].upper()} [{mm['hallucination_type']}]: "
                  f"{mm['message'][:80]}")

        results.append({
            "pair_id":               pair["pair_id"],
            "requirement":           req,
            "p1_label":              p1_lbl,
            "p1_hallucination_type": p1_ht,
            "semantic_result":       imr.to_dict(),
            "our_prediction":        our_pred,
            "agrees_with_p1":        agreement,
        })

    overall_acc = correct_preds / len(pairs) * 100

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "metadata": {
            "created_at":            datetime.now(UTC).isoformat(),
            "embedding_backend":     validator._backend,
            "total":                 len(results),
            "semantic_matches":      match_ct,
            "semantic_mismatches":   mismatch_ct,
            "overall_agreement_pct": f"{overall_acc:.1f}%",
            "hallucination_type_breakdown": htype_counts,
            "per_category_agreement": {
                cat: f"{v['agreed']}/{v['total']} "
                     f"({v['agreed']/v['total']*100:.0f}%)"
                for cat, v in category_stats.items()
            }
        },
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  Backend:              {validator._backend}")
    print(f"  Overall agreement:    {overall_acc:.1f}%  "
          f"(was 76.9% with TF-IDF v1)")
    print(f"  Matches:              {match_ct}")
    print(f"  Mismatches:           {mismatch_ct}")
    print(f"\n  Hallucination types detected:")
    for ht, cnt in sorted(htype_counts.items(), key=lambda x: -x[1]):
        print(f"    {ht:<32} {cnt}")
    print(f"\n  Per-category agreement:")
    for cat, v in sorted(category_stats.items()):
        pct = v['agreed'] / v['total'] * 100
        bar = "█" * int(pct // 10)
        print(f"    {cat:<30} {v['agreed']:>2}/{v['total']:>2}  "
              f"({pct:>5.1f}%)  {bar}")
    print(f"\n  Saved to: {output_path}")
    print(f"{'='*68}\n")

    return output


if __name__ == "__main__":
    run_semantic_v3()