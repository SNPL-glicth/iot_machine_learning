"""Setup script for Weaviate schema in the IoT ML service.

Usage:
    python infrastructure/weaviate/setup.py --host http://localhost:8080
"""

import argparse
import json
import sys
from pathlib import Path

import requests


def load_schema() -> dict:
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_class(host: str, class_def: dict) -> bool:
    resp = requests.post(f"{host}/v1/schema", json=class_def)
    if resp.status_code == 200:
        print(f"  Created class: {class_def['class']}")
        return True
    if resp.status_code == 409:
        print(f"  Already exists: {class_def['class']}")
        return True
    print(f"  ERROR {resp.status_code} creating {class_def['class']}: {resp.text}")
    return False


def delete_class(host: str, class_name: str) -> bool:
    resp = requests.delete(f"{host}/v1/schema/{class_name}")
    if resp.status_code in (200, 204, 404):
        print(f"  Deleted class: {class_name}")
        return True
    print(f"  ERROR {resp.status_code} deleting {class_name}: {resp.text}")
    return False


def setup_weaviate(host: str, recreate: bool = False) -> bool:
    print(f"Setting up Weaviate at {host}")
    schema = load_schema()
    classes = schema.get("classes", [])

    if recreate:
        print("Recreate mode: deleting existing classes first...")
        for class_def in classes:
            delete_class(host, class_def["class"])

    print(f"Creating {len(classes)} classes...")
    ok = True
    for class_def in classes:
        if not create_class(host, class_def):
            ok = False

    if ok:
        print("Schema setup complete.")
    else:
        print("Schema setup had errors.", file=sys.stderr)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Set up Weaviate schema for IoT ML")
    parser.add_argument("--host", default="http://localhost:8080", help="Weaviate REST endpoint")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate classes")
    args = parser.parse_args()

    ok = setup_weaviate(args.host, args.recreate)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
