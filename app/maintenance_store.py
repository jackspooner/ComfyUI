"""Local durable fencing and non-executable prompt receipts.

The caller serializes all access with the native queue mutex. Nothing here
restores executable work or infers completion from an empty in-memory queue.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

try:
    import fcntl
except ImportError:  # Optional Linux-only operator facility.
    fcntl = None


MAX_RECEIPTS = 10000
MAX_OPERATIONS = 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
RETENTION_SECONDS = 7 * 86400


class MaintenanceError(Exception):
    def __init__(self, code, status=409, accepted=False):
        super().__init__(code)
        self.code = code
        self.status = status
        self.accepted = accepted


class PromptReplay(Exception):
    def __init__(self, receipt):
        self.receipt = receipt


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class MaintenanceStore:
    def __init__(self, directory, *, max_receipts=MAX_RECEIPTS):
        if type(max_receipts) is not int or max_receipts < 1:
            raise ValueError("maintenance max receipts must be a positive integer")
        self.max_receipts = max_receipts
        if fcntl is None:
            raise ValueError("maintenance requires local POSIX file locking")
        directory = Path(directory)
        if not directory.is_absolute():
            raise ValueError("maintenance directory must be absolute")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = directory.lstat()
        if directory.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise ValueError("maintenance directory must be owned by this user with mode 0700")
        self.lock_file = open(directory / "owner.lock", "a+b")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(directory / "maintenance.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            self.close()
            raise ValueError("unsupported maintenance store version")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                revision INTEGER NOT NULL, epoch INTEGER NOT NULL,
                active TEXT, generation TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, target TEXT NOT NULL,
                epoch INTEGER NOT NULL, generation TEXT NOT NULL,
                phase TEXT NOT NULL, release_request TEXT);
            CREATE TABLE IF NOT EXISTS receipts (
                id TEXT PRIMARY KEY, digest TEXT NOT NULL, number REAL NOT NULL,
                state TEXT NOT NULL, created REAL NOT NULL, terminal REAL,
                history TEXT, visible INTEGER NOT NULL DEFAULT 1);
            CREATE INDEX IF NOT EXISTS receipts_terminal ON receipts(terminal);
        """)
        self.generation = str(uuid.uuid4())
        self.invocation = os.environ.get("INVOCATION_ID") or self.generation
        self.failure = None
        with self.db:
            if version in (0, 1) and "visible" not in {row["name"] for row in self.db.execute("PRAGMA table_info(receipts)")}:
                self.db.execute("ALTER TABLE receipts ADD COLUMN visible INTEGER NOT NULL DEFAULT 1")
            self.db.execute("PRAGMA user_version=2")
            self.db.execute("INSERT OR IGNORE INTO state VALUES (1,0,0,NULL,?)", (self.generation,))
            self.db.execute("UPDATE state SET generation=?,revision=revision+1 WHERE singleton=1", (self.generation,))
            self.db.execute("UPDATE receipts SET state='interrupted_unknown' WHERE state='accepted'")

    def close(self):
        self.db.close()
        self.lock_file.close()

    @contextlib.contextmanager
    def transaction(self):
        try:
            with self.db:
                yield
        except sqlite3.Error as exc:
            self.failure = "receipt_store_failure"
            raise MaintenanceError(self.failure, 503, accepted=None) from exc

    def read(self, sql, parameters=()):
        try:
            return self.db.execute(sql, parameters).fetchall()
        except sqlite3.Error as exc:
            self.failure = "receipt_store_failure"
            raise MaintenanceError(self.failure, 503, accepted=None) from exc

    def state(self):
        return dict(self.read("SELECT * FROM state WHERE singleton=1")[0])

    def probe_write(self):
        with self.transaction():
            self.db.execute("UPDATE state SET generation=? WHERE singleton=1", (self.generation,))

    def check_admission(self):
        if self.failure:
            raise MaintenanceError(self.failure, 503)
        if self.state()["active"]:
            raise MaintenanceError("maintenance_fenced", 503)
        if self.unresolved():
            raise MaintenanceError("interrupted_work_unresolved", 503)

    def unresolved(self):
        return self.read("SELECT count(*) FROM receipts WHERE state='interrupted_unknown'")[0][0]

    def accept(self, item):
        # Hash execution-affecting content, not queue priority or server timestamps.
        extra = {k: v for k, v in item[3].items() if k not in ("create_time", "client_id")}
        try:
            digest = hashlib.sha256(encode([item[2], extra, item[4], item[5:]]).encode()).hexdigest()
        except (TypeError, ValueError) as exc:
            raise MaintenanceError("invalid_receipt_input", 400) from exc
        old = self.receipt(item[1])
        if old:
            if old["digest"] != digest:
                raise MaintenanceError("prompt_id_conflict")
            raise PromptReplay(old)
        self.check_admission()
        with self.transaction():
            self.db.execute("DELETE FROM receipts WHERE terminal < ?", (time.time() - RETENTION_SECONDS,))
            if self.db.execute("SELECT count(*) FROM receipts").fetchone()[0] >= self.max_receipts:
                raise MaintenanceError("receipt_capacity", 503)
            self.db.execute("INSERT INTO receipts (id,digest,number,state,created,terminal,history,visible) VALUES (?,?,?,'accepted',?,NULL,NULL,1)",
                            (item[1], digest, item[0], time.time()))

    def complete(self, prompt_id, history):
        try:
            payload = encode(history)
            if len(payload.encode()) > MAX_RECEIPT_BYTES:
                raise ValueError("receipt too large")
        except (TypeError, ValueError) as exc:
            self.failure = "receipt_serialization_failure"
            raise MaintenanceError(self.failure, 503, accepted=True) from exc
        with self.transaction():
            changed = self.db.execute("UPDATE receipts SET state='terminal',terminal=?,history=? WHERE id=? AND state='accepted'",
                                      (time.time(), payload, prompt_id)).rowcount
            if changed != 1:
                self.failure = "receipt_missing"
                raise MaintenanceError(self.failure, 503, accepted=None)

    def receipt(self, prompt_id):
        rows = self.read("SELECT * FROM receipts WHERE id=?", (prompt_id,))
        if not rows:
            return None
        receipt = dict(rows[0])
        receipt["history"] = json.loads(receipt["history"]) if receipt["history"] else None
        return receipt

    def histories(self):
        return {row["id"]: json.loads(row["history"]) for row in
                self.read("SELECT id,history FROM receipts WHERE state='terminal' AND visible=1 ORDER BY terminal")}

    def hide_history(self, prompt_id=None):
        with self.transaction():
            if prompt_id is None:
                self.db.execute("UPDATE receipts SET visible=0 WHERE state='terminal'")
            else:
                self.db.execute("UPDATE receipts SET visible=0 WHERE id=? AND state='terminal'", (prompt_id,))

    def status(self, pending, running, validating, finishing=0, publishing=0, unpublished_receipts=0):
        state = self.state()
        active = state["active"]
        operation = self.read("SELECT * FROM operations WHERE id=?", (active,))[0] if active else None
        unknown = self.unresolved()
        accepted = self.read("SELECT count(*) FROM receipts WHERE state='accepted'")[0][0]
        blockers = []
        if self.failure:
            blockers.append(self.failure)
        if unknown:
            blockers.append("interrupted_work_unresolved")
        if accepted != pending + running + unpublished_receipts:
            blockers.append("queue_receipt_mismatch")
        ready = bool(active and not (pending or running or validating or finishing or publishing or accepted or blockers))
        return {"schemaVersion": 1, "revision": state["revision"], "epoch": state["epoch"],
                "generation": self.generation, "invocationId": self.invocation,
                "operationId": active, "ownerId": operation["owner"] if operation else None,
                "targetRevision": operation["target"] if operation else None,
                "phase": ("blocked" if blockers else "ready" if ready else "draining") if active else "open",
                "accepting": not active and not blockers, "safeToStop": ready,
                "counts": {"pending": pending, "running": running, "validating": validating,
                           "finishing": finishing, "publishing": publishing, "accepted": accepted,
                           "interruptedUnknown": unknown}, "blockers": blockers}

    def check_operator(self, request):
        state = self.state()
        operations = self.read("SELECT * FROM operations WHERE id=?", (request["operationId"],))
        operation = operations[0] if operations else None
        if not operation or state["active"] != request["operationId"] or operation["owner"] != request["ownerId"]:
            raise MaintenanceError("operation_conflict")
        if request["expectedRevision"] != state["revision"] or request["expectedGeneration"] != self.generation:
            raise MaintenanceError("revision_conflict")
        return operation

    def start(self, request):
        state = self.state()
        operations = self.read("SELECT * FROM operations WHERE id=?", (request["operationId"],))
        old = operations[0] if operations else None
        if old:
            if (old["owner"], old["target"]) != (request["ownerId"], request["targetRevision"]):
                raise MaintenanceError("operation_identity_conflict")
            # A completed identity is never re-acquired, even after a lost response.
            if state["active"] != old["id"]:
                raise MaintenanceError("operation_already_released")
            return
        if state["active"]:
            raise MaintenanceError("operation_conflict")
        if request["expectedRevision"] != state["revision"]:
            raise MaintenanceError("revision_conflict")
        with self.transaction():
            if self.db.execute("SELECT count(*) FROM operations").fetchone()[0] >= MAX_OPERATIONS:
                raise MaintenanceError("operation_capacity", 503)
            self.db.execute("INSERT INTO operations VALUES (?,?,?,?,?,'fenced',NULL)",
                            (request["operationId"], request["ownerId"], request["targetRevision"],
                             state["epoch"] + 1, self.generation))
            self.db.execute("UPDATE state SET active=?,epoch=epoch+1,revision=revision+1 WHERE singleton=1", (request["operationId"],))

    def release(self, action, request, ready):
        operations = self.read("SELECT * FROM operations WHERE id=?", (request["operationId"],))
        old = operations[0] if operations else None
        replay = encode([action, request])
        if old and old["release_request"] == replay:
            return
        self.check_operator(request)
        if action == "abort" and old["generation"] != self.generation:
            raise MaintenanceError("abort_after_restart_forbidden")
        if not ready:
            raise MaintenanceError("maintenance_not_ready")
        with self.transaction():
            self.db.execute("UPDATE operations SET phase=?,release_request=? WHERE id=?", (action, replay, old["id"]))
            self.db.execute("UPDATE state SET active=NULL,revision=revision+1 WHERE singleton=1")
