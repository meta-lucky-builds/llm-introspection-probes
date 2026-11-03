"""What makes rows from different sessions comparable. A store holds one identity only."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def code_hash() -> str:
    """Hash of this package's source. Line endings are normalized so a checkout's OS does not matter."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def run_identity(
    config_hash: str,
    data_sha256: str,
    model_commit: str,
    dtype: str,
    device_type: str,
    versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "config_hash": config_hash,
        "data_sha256": data_sha256,
        "code_hash": code_hash(),
        "model_commit": model_commit,
        "dtype": dtype,
        "device_type": device_type,
        "versions": versions,
    }


def identity_differences(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """Names of the identity fields that differ, with library versions compared one by one."""
    differences = []
    for name in sorted(set(first) | set(second)):
        if name == "versions":
            continue
        if first.get(name) != second.get(name):
            differences.append(name)
    first_versions = first.get("versions") or {}
    second_versions = second.get("versions") or {}
    for name in sorted(set(first_versions) | set(second_versions)):
        if first_versions.get(name) != second_versions.get(name):
            differences.append(f"versions.{name}")
    return differences
