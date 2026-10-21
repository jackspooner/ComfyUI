"""Durable, single-owner archive cache. This module never imports torch."""
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time

import yaml
import safetensors

CATEGORIES = {"checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "controlnet", "clip_vision"}
EXTENSIONS = {".safetensors", ".sft"}
CHUNK = 8 * 1024 * 1024


class CacheError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def fingerprint(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise CacheError("CONFIG_INVALID", "Expected a YAML mapping.")
    allowed = {"archive_root", "archive_uuid", "cache_root", "max_bytes", "reserve_bytes", "idle_seconds"}
    if set(config) - allowed or not {"archive_root", "archive_uuid", "cache_root"} <= set(config):
        raise CacheError("CONFIG_INVALID", "Specify archive_root, archive_uuid and cache_root; unknown fields are rejected.")
    for key in ("archive_root", "cache_root"):
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise CacheError("CONFIG_INVALID", f"{key} must be an absolute path.")
        config[key] = str(Path(config[key]).resolve())
    if not isinstance(config["archive_uuid"], str) or not config["archive_uuid"]:
        raise CacheError("CONFIG_INVALID", "archive_uuid must name the source filesystem UUID (findmnt -o UUID).")
    for key, default in (("max_bytes", 100 * 1024**3), ("reserve_bytes", 100 * 1024**3), ("idle_seconds", 1800)):
        value = config.setdefault(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < (0 if key == "reserve_bytes" else 1):
            raise CacheError("CONFIG_INVALID", f"{key} must be a finite positive number (reserve may be zero).")
    archive, cache = Path(config["archive_root"]), Path(config["cache_root"])
    if archive == cache or archive in cache.parents or cache in archive.parents:
        raise CacheError("CONFIG_INVALID", "Archive and cache directories must not overlap.")
    return config


def check_archive(config):
    root = Path(config["archive_root"])
    if not root.is_dir():
        raise CacheError("ARCHIVE_UNAVAILABLE", "Mount the configured model archive before loading models.")
    result = subprocess.run(["findmnt", "-n", "-o", "UUID", "--target", str(root)], capture_output=True, text=True, check=False)
    if result.returncode or result.stdout.strip().lower() != config["archive_uuid"].lower():
        raise CacheError("ARCHIVE_UNAVAILABLE", "Archive filesystem UUID does not match configuration.")
    return root


def source_path(config, category, name):
    if category not in CATEGORIES or not isinstance(name, str):
        raise CacheError("MODEL_INVALID", "Unsupported archive category or filename.")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in EXTENSIONS:
        raise CacheError("MODEL_INVALID", "Expected a relative safetensors filename.")
    root = check_archive(config)
    path = root / category / relative
    # No symlinks in the admitted archive chain, including category directories.
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise CacheError("MODEL_INVALID", "Archive symlinks are not supported; use regular files.")
    if not path.is_file():
        raise CacheError("MODEL_MISSING", f"Archive model is unavailable: {category}/{name}")
    return path


def open_database(root):
    db = sqlite3.connect(root / "state.sqlite3", timeout=30, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


def mapped_files():
    """Include open handles as well as mmap references, whose fd may be closed."""
    identities = set()
    with open("/proc/self/maps", encoding="utf-8") as f:
        for line in f:
            fields = line.split(None, 5)
            if len(fields) >= 5 and int(fields[4]):
                major, minor = (int(v, 16) for v in fields[3].split(":"))
                identities.add((os.makedev(major, minor), int(fields[4])))
    for path in Path("/proc/self/fd").iterdir():
        try:
            s = path.stat()
        except FileNotFoundError:  # The directory enumeration itself opens a transient fd.
            continue
        identities.add((s.st_dev, s.st_ino))
    return identities


class ModelCache:
    def __init__(self, config):
        if sys.platform != "linux":
            raise CacheError("PLATFORM_UNSUPPORTED", "The optional model cache requires Linux.")
        import fcntl  # Optional Linux feature; ordinary ComfyUI remains portable.
        self.config = config
        self.root = Path(config["cache_root"])
        self.archive = Path(config["archive_root"])
        self.mutex = threading.RLock()
        self.verified = {}
        self.monotonic_used = {}
        self.owner = None
        self.db = None
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or self.root.stat().st_uid != os.getuid() or stat.S_IMODE(self.root.stat().st_mode) & 0o077:
            raise CacheError("CACHE_UNSAFE", "Use a dedicated cache directory owned by you with mode 0700.")
        self.owner = (self.root / "owner.lock").open("a+")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise CacheError("CACHE_BUSY", "Another ComfyUI process owns this cache.") from None
        try:
            marker = self.root / "owner.json"
            identity = {"schema": 1, "archive_root": str(self.archive), "archive_uuid": config["archive_uuid"]}
            if marker.exists():
                if json.loads(marker.read_text()) != identity:
                    raise CacheError("CACHE_IDENTITY", "Cache belongs to a different archive or schema.")
            else:
                if set(p.name for p in self.root.iterdir()) - {"owner.lock"}:
                    raise CacheError("CACHE_UNSAFE", "New cache directory must be empty.")
                with marker.open("x") as f:
                    json.dump(identity, f)
                    f.flush()
                    os.fsync(f.fileno())
                sync_directory(self.root)
            self.files = self.root / "files"
            self.files.mkdir(mode=0o700, exist_ok=True)
            if self.files.is_symlink():
                raise CacheError("CACHE_UNSAFE", "Cache files directory must not be a symlink.")
            self.db = open_database(self.root)
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY, source TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    size INTEGER NOT NULL, digest TEXT, state TEXT NOT NULL,
                    last_used REAL NOT NULL, active INTEGER NOT NULL DEFAULT 0,
                    pinned INTEGER NOT NULL DEFAULT 0, evict INTEGER NOT NULL DEFAULT 0,
                    copied INTEGER NOT NULL DEFAULT 0, reason TEXT);
                CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY, operation TEXT NOT NULL, argument TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'pending', result TEXT);
            """)
            self.recover()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.owner is not None:
            self.owner.close()
            self.owner = None

    def path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise CacheError("STATE_INVALID", "Invalid cache entry identity.")
        return self.files / (key + ".safetensors")

    def recover(self):
        with self.db:
            for row in self.db.execute("SELECT * FROM entries").fetchall():
                path = self.path(row["key"])
                partial = path.with_suffix(".partial")
                if partial.exists():
                    partial.unlink()
                if row["state"] != "ready":
                    path.unlink(missing_ok=True)
                    self.db.execute("DELETE FROM entries WHERE key=?", (row["key"],))
                elif not path.is_file() or path.is_symlink() or path.stat().st_size != row["size"]:
                    self.db.execute("UPDATE entries SET reason='Cached file missing or invalid', evict=1 WHERE key=?", (row["key"],))
                elif row["active"] or row["last_used"] > time.time():
                    # An interrupted execution or clock rollback gets a fresh idle interval.
                    self.db.execute("UPDATE entries SET active=0,last_used=? WHERE key=?", (time.time(), row["key"]))
            self.db.execute("UPDATE commands SET state='failed',result='Interrupted; inspect state before retrying' WHERE state='running'")
        sync_directory(self.files)

    def stage(self, source, progress=None):
        source = Path(source).absolute()
        try:
            relative = source.relative_to(self.archive)
        except ValueError:
            return str(source), None
        if len(relative.parts) < 2:
            raise CacheError("MODEL_INVALID", "Expected an archive category and filename.")
        source = source_path(self.config, relative.parts[0], str(Path(*relative.parts[1:])))
        with self.mutex:
            before = fingerprint(source)
            identity = json.dumps([str(source), before], separators=(",", ":"))
            key = hashlib.sha256(identity.encode()).hexdigest()
            path = self.path(key)
            row = self.db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
            if row is not None and row["state"] == "ready":
                if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size"]:
                    raise CacheError("CACHE_CORRUPT", "Cached model changed; request eviction and retry.")
                if self.verified.get(key) != fingerprint(path) and digest(path) != row["digest"]:
                    raise CacheError("CACHE_CORRUPT", "Cached model checksum failed; request eviction and retry.")
                self.verified[key] = fingerprint(path)
                self.touch({key})
                return str(path), key
            size = before[2]
            used = self.db.execute("SELECT COALESCE(SUM(size),0) FROM entries").fetchone()[0]
            if used + size > self.config["max_bytes"] or shutil.disk_usage(self.root).free - size < self.config["reserve_bytes"]:
                raise CacheError("CACHE_FULL", "Not enough cache capacity or reserved free space. Evict unused entries or increase capacity.")
            with self.db:
                self.db.execute("INSERT INTO entries(key,source,fingerprint,size,state,last_used) VALUES(?,?,?,?, 'copying',?)", (key, str(source), json.dumps(before), size, time.time()))
            partial = path.with_suffix(".partial")
            try:
                copied, checksum = 0, hashlib.sha256()
                logging.info("Staging archive model: %s (%d bytes)", relative, size)
                with source.open("rb") as src, partial.open("xb") as dst:
                    while block := src.read(CHUNK):
                        if shutil.disk_usage(self.root).free - len(block) < self.config["reserve_bytes"]:
                            raise CacheError("CACHE_FULL", "Free-space reserve reached while copying.")
                        dst.write(block)
                        checksum.update(block)
                        copied += len(block)
                        with self.db:
                            self.db.execute("UPDATE entries SET copied=? WHERE key=?", (copied, key))
                        if progress:
                            progress(copied, size)
                    dst.flush()
                    os.fsync(dst.fileno())
                if copied != size or fingerprint(source) != before:
                    raise CacheError("SOURCE_CHANGED", "Source changed during copying; retry after it is stable.")
                if digest(partial) != checksum.hexdigest():
                    raise CacheError("COPY_CORRUPT", "Copied model checksum failed.")
                try:
                    with safetensors.safe_open(partial, framework="np"):
                        pass
                except safetensors.SafetensorError as error:
                    raise CacheError("MODEL_CORRUPT", "Invalid or incomplete safetensors model.") from error
                partial.chmod(0o400)
                os.replace(partial, path)
                sync_directory(self.files)
                with self.db:
                    self.db.execute("UPDATE entries SET state='ready',digest=? WHERE key=?", (checksum.hexdigest(), key))
                self.verified[key] = fingerprint(path)
                self.touch({key})
                return str(path), key
            except BaseException:
                partial.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
                with self.db:
                    self.db.execute("DELETE FROM entries WHERE key=?", (key,))
                raise

    def touch(self, keys, finished=False):
        with self.mutex, self.db:
            for key in keys:
                self.db.execute("UPDATE entries SET last_used=?,active=?,evict=0,reason=NULL WHERE key=?", (time.time(), 0 if finished else 1, key))
                self.monotonic_used[key] = time.monotonic()

    def expired(self):
        with self.mutex:
            rows = self.db.execute("SELECT * FROM entries WHERE state='ready' AND pinned=0").fetchall()
            result = set()
            for row in rows:
                elapsed = time.monotonic() - self.monotonic_used[row["key"]] if row["key"] in self.monotonic_used else max(0, time.time() - row["last_used"])
                if row["evict"] or elapsed >= self.config["idle_seconds"]:
                    result.add(row["key"])
            return result

    def changed(self):
        """Invalidate cached execution results before reusing a replaced source."""
        with self.mutex:
            rows = self.db.execute("SELECT * FROM entries WHERE state='ready'").fetchall()
            try:
                check_archive(self.config)
            except CacheError:
                return {row["key"] for row in rows}
            changed = set()
            for row in rows:
                try:
                    if json.loads(row["fingerprint"]) != fingerprint(Path(row["source"])):
                        changed.add(row["key"])
                except OSError:
                    changed.add(row["key"])
            return changed

    def evict(self, keys):
        with self.mutex:
            opened = mapped_files()
            for key in keys:
                row = self.db.execute("SELECT * FROM entries WHERE key=?", (key,)).fetchone()
                if row is None or row["pinned"]:
                    continue
                path = self.path(key)
                if path.exists() and (path.stat().st_dev, path.stat().st_ino) in opened:
                    with self.db:
                        self.db.execute("UPDATE entries SET reason='Still mapped or open in ComfyUI' WHERE key=?", (key,))
                    continue
                with self.db:
                    self.db.execute("UPDATE entries SET state='deleting' WHERE key=?", (key,))
                path.unlink(missing_ok=True)
                sync_directory(self.files)
                with self.db:
                    self.db.execute("DELETE FROM entries WHERE key=?", (key,))
                self.verified.pop(key, None)
                self.monotonic_used.pop(key, None)
                logging.info("Evicted cached model: %s", row["source"])

    def commands(self):
        with self.mutex:
            rows = self.db.execute("SELECT * FROM commands WHERE state='pending' ORDER BY id").fetchall()
            for row in rows:
                with self.db:
                    self.db.execute("UPDATE commands SET state='running' WHERE id=?", (row["id"],))
                try:
                    argument = row["argument"]
                    if row["operation"] == "prefetch":
                        category, name = argument.split("/", 1)
                        _, key = self.stage(source_path(self.config, category, name))
                        self.touch({key}, finished=True)
                    else:
                        entry = self.db.execute("SELECT * FROM entries WHERE key=?", (argument,)).fetchone()
                        if entry is None:
                            raise CacheError("ENTRY_MISSING", "Entry no longer exists; refresh status.")
                        if row["operation"] == "evict" and entry["pinned"]:
                            raise CacheError("ENTRY_PINNED", "Unpin the entry before eviction.")
                        with self.db:
                            if row["operation"] in ("pin", "unpin"):
                                self.db.execute("UPDATE entries SET pinned=? WHERE key=?", (int(row["operation"] == "pin"), argument))
                            elif row["operation"] == "evict":
                                self.db.execute("UPDATE entries SET evict=1 WHERE key=?", (argument,))
                            else:
                                raise CacheError("COMMAND_INVALID", "Unknown operation.")
                    state, result = "done", "Accepted; inspect entry status for deferred eviction."
                except (CacheError, OSError, ValueError) as error:
                    state, result = "failed", str(error)
                with self.db:
                    self.db.execute("UPDATE commands SET state=?,result=? WHERE id=?", (state, result, row["id"]))
