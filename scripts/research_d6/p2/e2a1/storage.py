"""Atomic, checksum-verified NPY bundles for E2-A1 outputs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_bundle(final_dir: Path, expected_identity: dict[str, Any],
                   expected_metadata: dict[str, Any], expected_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    record_path = final_dir / "record.json"
    if not record_path.is_file():
        raise RuntimeError(f"Existing bundle is incomplete (record missing): {final_dir.name}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("identity") != expected_identity or record.get("metadata") != expected_metadata:
        raise RuntimeError(f"Existing bundle identity mismatch; refusing overwrite: {final_dir.name}")
    specs = record.get("arrays")
    if not isinstance(specs, dict) or set(specs) != set(expected_arrays):
        raise RuntimeError(f"Existing bundle array set mismatch: {final_dir.name}")
    for name, expected in expected_arrays.items():
        spec = specs[name]
        path = final_dir / spec.get("file", "")
        if not path.is_file() or sha256_file(path) != spec.get("file_sha256"):
            raise RuntimeError(f"Existing bundle file SHA mismatch: {final_dir.name}/{name}")
        actual = np.load(path, allow_pickle=False)
        if (actual.dtype != expected.dtype or actual.shape != expected.shape
                or not np.isfinite(actual).all()
                or array_sha256(actual) != spec.get("array_sha256")):
            raise RuntimeError(f"Existing bundle array mismatch: {final_dir.name}/{name}")
    return record


def write_bundle_atomic(final_dir: Path, arrays: dict[str, np.ndarray],
                        identity: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Write one image's complete array bundle atomically; verify/reuse exact matches.

    Returns ``(record, reused_existing)``. Any same-name mismatch blocks. The
    temporary directory is created beside the destination so the final rename
    stays on one filesystem.
    """
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        return _verify_bundle(final_dir, identity, metadata, arrays), True
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", suffix=".tmp", dir=final_dir.parent))
    try:
        specs: dict[str, Any] = {}
        for name in sorted(arrays):
            if not name.replace("_", "").isalnum():
                raise ValueError(f"Unsafe array key: {name}")
            array = np.ascontiguousarray(arrays[name])
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError(f"Array is object-typed or nonfinite: {name}")
            filename = f"{name}.npy"
            path = temp_dir / filename
            _write_npy(path, array)
            specs[name] = {
                "file": filename,
                "file_sha256": sha256_file(path),
                "array_sha256": array_sha256(array),
                "dtype": array.dtype.name,
                "shape": list(array.shape),
            }
        record = {"identity": identity, "metadata": metadata, "arrays": specs}
        record_path = temp_dir / "record.json"
        with record_path.open("xb") as stream:
            stream.write(stable_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temp_dir, final_dir)
        except FileExistsError:
            # A concurrent writer is not expected, but an exact completed item
            # can safely be reused; otherwise identity verification blocks.
            shutil.rmtree(temp_dir, ignore_errors=True)
            return _verify_bundle(final_dir, identity, metadata, arrays), True
        return record, False
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def write_manifest_once(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> str:
    import csv
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"Existing manifest differs; refusing overwrite: {path.name}")
    else:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temp_name, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise RuntimeError(f"Concurrent manifest differs; refusing overwrite: {path.name}")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return hashlib.sha256(payload).hexdigest()
