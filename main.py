"""
main.py — Week 1 demo runner
Loads the seed dataset and runs the syntax validator on all 20 rules.
Prints a full report to the terminal.

Usage:
    python main.py
    python main.py --file data/seed_rules.json
    python main.py --rule '{"rule_id":"x","action":"allow","protocol":"tcp","source":{"ip":"any"},"destination":{"ip":"10.0.0.1","port":80}}'
"""

import json
import sys
import os
import argparse

# Make sure we can import from validator/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validator.syntax_validator import SyntaxValidator, print_report


def main():
    parser = argparse.ArgumentParser(description="Firewall Rule Syntax Validator — Week 1")
    parser.add_argument("--file", default="data/seed_rules.json",
                        help="Path to JSON file containing a list of rules")
    parser.add_argument("--rule", default=None,
                        help="Single rule as a JSON string")
    parser.add_argument("--schema", default="schemas/firewall_rule_schema.json",
                        help="Path to the JSON schema file")
    args = parser.parse_args()

    validator = SyntaxValidator(schema_path=args.schema)

    if args.rule:
        rule = json.loads(args.rule)
        results = [validator.validate_rule(rule)]
    else:
        with open(args.file) as f:
            rules = json.load(f)
        results = validator.validate_batch(rules)

    print_report(results)

    # Summary counts
    total   = len(results)
    passed  = sum(1 for r in results if r.is_valid)
    failed  = total - passed
    print(f"Final: {passed}/{total} rules passed syntax validation.\n")

    # Return exit code 1 if any rules failed (useful in CI pipelines)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
