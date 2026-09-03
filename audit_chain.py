"""
audit_chain.py -- Hash-chained, byte-reproducible audit logging.

Provides cryptographic tamper-evidence and deterministic serialization
for Abuse-Ring Sentinel audit logs.

Properties:
  - Canonical JSON: sorted keys, compact separators (",", ":"), UTF-8 encoding.
  - Genesis entry (index 0) uses "0" * 64 as prev_hash.
  - Each entry connects cryptographically to the previous entry's entry_hash.
  - Zero wall-clock timestamps or nondeterministic structures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List


GENESIS_PREV_HASH: str = "0" * 64


def canonical_json(obj: Any) -> str:
    """
    Deterministically serializes any JSON-compatible object to a compact string.
    This is the ONLY serializer used across the chain for hashing and writing.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    """Computes standard SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def compute_entry_hash(entry_index: int, payload: dict, prev_hash: str) -> str:
    """Computes entry_hash from entry content (index, payload, prev_hash)."""
    body = {
        "entry_index": entry_index,
        "payload": payload,
        "prev_hash": prev_hash,
    }
    return sha256_hex(canonical_json(body))


def build_chain(payloads: List[dict]) -> List[dict]:
    """
    Constructs a hash-chained sequence of audit entries from risk output payloads.

    Parameters
    ----------
    payloads : List[dict]
        Ordered list of cluster risk dictionaries (from ClusterRiskOutput.to_json_dict()).

    Returns
    -------
    List of chain entries with structure:
      {
        "entry_index": int,
        "payload": dict,
        "prev_hash": str,
        "entry_hash": str,
      }
    """
    chain: List[dict] = []
    prev_hash = GENESIS_PREV_HASH

    for i, payload in enumerate(payloads):
        entry_hash = compute_entry_hash(i, payload, prev_hash)
        entry = {
            "entry_index": i,
            "payload": payload,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }
        chain.append(entry)
        prev_hash = entry_hash

    return chain


def write_chain(chain: List[dict], path: str | Path = "audit_log.jsonl") -> None:
    """
    Writes one canonical_json(entry) per line to the target path.
    Guaranteed byte-identical output given the same chain.
    """
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for entry in chain:
            f.write(canonical_json(entry) + "\n")
