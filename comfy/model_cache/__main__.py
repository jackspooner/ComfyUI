"""Local control client. Commands are durably queued for the ComfyUI worker."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .store import CacheError, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("operation", choices=("status", "prefetch", "pin", "unpin", "evict"))
    parser.add_argument("argument", nargs="?", help="category/relative.safetensors for prefetch; entry key otherwise")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        path = Path(config["cache_root"]) / "state.sqlite3"
        with sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=30) as db:
            db.row_factory = sqlite3.Row
            if args.operation == "status":
                entries = [dict(r) for r in db.execute("SELECT * FROM entries ORDER BY last_used DESC")]
                commands = [dict(r) for r in db.execute("SELECT * FROM commands ORDER BY id DESC LIMIT 20")]
                result = {"entries": entries, "commands": commands, "bytes": sum(r["size"] for r in entries), "max_bytes": config["max_bytes"], "idle_seconds": config["idle_seconds"]}
            else:
                if not args.argument:
                    parser.error("This operation requires an argument.")
                cursor = db.execute("INSERT INTO commands(operation,argument) VALUES(?,?)", (args.operation, args.argument))
                result = {"command_id": cursor.lastrowid, "state": "pending", "message": "Runs at the next idle worker boundary; requires ComfyUI with this configuration."}
        print(json.dumps(result, indent=2))
        return 0
    except (CacheError, OSError, sqlite3.Error) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
