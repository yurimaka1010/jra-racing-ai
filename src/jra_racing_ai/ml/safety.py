"""Fixed diagnostic codes and local-only, exclusive artifact I/O."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from ..core import SafeAuditError
from ..generate_report import PROJECT_ROOT, _has_link, _local_disk, validate_output_dir
from ..reconciliation_local import _cloud_or_reparse

ML_ROOT = Path(r"C:\Users\acbd5\AppData\Local\JraRacingAI\ml-v01")
MAX_ROWS = 500_000
MAX_FILE_BYTES = 512 * 1024 * 1024
REASONS = frozenset({
    "invalid_input", "invalid_mapping", "mapping_not_reviewed", "unsafe_path",
    "input_limit", "duplicate_entry", "invalid_identity", "invalid_date",
    "insufficient_history", "invalid_split", "missing_library", "training_failed",
    "invalid_model", "artifact_exists", "artifact_invalid", "io_failed",
    "needs_authentication", "explicit_execution_required", "query_limit",
    "query_timeout", "database_error", "invalid_schema", "invalid_odds",
    "forward_cutoff", "ledger_tampered", "outcomes_incomplete", "internal_error",
})


class MLFailure(SafeAuditError):
    def __init__(self, code="internal_error"):
        self.code = code if code in REASONS else "internal_error"
        # This class owns the ML fixed-code allowlist. A read-only session must
        # preserve it rather than reclassifying application failures as DB errors.
        Exception.__init__(self, self.code)


def local_path(path: Path, *, exists=False, root=None) -> Path:
    """Lexical containment before filesystem probes; reject sync/reparse paths."""
    root = ML_ROOT if root is None else root
    try:
        root = validate_output_dir(root, real=True, approved_root=root,
                                   project_root=PROJECT_ROOT)
        if (not path.is_absolute() or ".." in path.parts or not _local_disk(path)
                or not (path == root or root in path.parents)):
            raise MLFailure("unsafe_path")
        if _has_link(path) or _cloud_or_reparse(path):
            raise MLFailure("unsafe_path")
        resolved = path.resolve()
        if not (resolved == root or root in resolved.parents):
            raise MLFailure("unsafe_path")
        if exists:
            if (not resolved.is_file() or resolved.stat().st_nlink != 1
                    or resolved.stat().st_size > MAX_FILE_BYTES):
                raise MLFailure("input_limit")
        return resolved
    except MLFailure:
        raise
    except Exception:
        raise MLFailure("unsafe_path") from None


def new_artifact(kind: str, *, root=None) -> Path:
    if kind not in ("schema", "datasets", "models", "predictions", "evaluations", "demo"):
        raise MLFailure("invalid_input")
    root = ML_ROOT if root is None else root
    path = local_path(root / kind / uuid4().hex, root=root)
    try:
        path.mkdir(parents=True, exist_ok=False)
        return path
    except Exception:
        raise MLFailure("io_failed") from None


def write_json(path: Path, value, *, root=None):
    path = local_path(path, root=root)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise MLFailure("artifact_exists") from None
    except Exception:
        raise MLFailure("io_failed") from None


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise MLFailure("invalid_input")
        value[key] = item
    return value


def read_json(path: Path, *, root=None):
    path = local_path(path, exists=True, root=root)
    try:
        with path.open(encoding="utf-8-sig") as stream:
            return json.load(stream, object_pairs_hook=_unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(MLFailure("invalid_input")))
    except MLFailure:
        raise
    except Exception:
        raise MLFailure("invalid_input") from None


def digest(path: Path, *, root=None) -> str:
    path = local_path(path, exists=True, root=root)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", value):
        raise MLFailure("invalid_mapping")
    return '"' + value + '"'
