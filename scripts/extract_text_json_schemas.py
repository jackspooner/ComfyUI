#!/usr/bin/env python3
"""Extract distinct schema-only fixtures from an explicitly supplied WF5 snapshot."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    schemas = {}

    def visit(value):
        if isinstance(value, dict):
            if value.get("typeId") == "router.text" and "responseSchema" in value.get("config", {}):
                schema = value["config"]["responseSchema"]
                encoded = json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode()
                schemas[hashlib.sha256(encoded).hexdigest()] = schema
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(json.loads(args.snapshot.read_text()))
    args.destination.mkdir(parents=True, exist_ok=True)
    for digest, schema in schemas.items():
        path = args.destination / (digest + ".json")
        with path.open("x") as output:
            json.dump(schema, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(f"Extracted {len(schemas)} schema-only fixtures; no prompts, paths or media copied.")


if __name__ == "__main__":
    main()
