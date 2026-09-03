"""
verify_audit_log.py -- Standalone verification engine for audit_log.jsonl.

Verifies:
  - Unbroken cryptographic hash chain from genesis to tail.
  - Genesis prev_hash equals "0" * 64.
  - Every entry's stored entry_hash matches recomputed SHA-256(canonical_json(body)).
  - Every entry's prev_hash matches the previous entry's actual entry_hash.
  - Contiguous 0-based entry_index sequence (0, 1, ..., N-1).
  - Handles empty audit log gracefully without crashing.
  - Pinpoints the exact entry_index and mismatch details on any tampering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple, Optional

from audit_chain import GENESIS_PREV_HASH, compute_entry_hash


def verify_audit_log(path: str | Path = "audit_log.jsonl") -> Tuple[bool, str, Optional[int]]:
    """
    Verifies the integrity of an audit log file.

    Returns
    -------
    Tuple of (is_valid: bool, message: str, broken_index: Optional[int])
    """
    p = Path(path)
    if not p.exists():
        return False, f"ERROR: File not found: {p}", None

    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not lines:
        return True, "OK: 0 entries, chain intact", None

    prev_expected_hash = GENESIS_PREV_HASH

    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            return False, f"TAMPER DETECTED at line {i}: invalid JSON syntax ({exc})", i

        if not isinstance(entry, dict):
            return False, f"TAMPER DETECTED at line {i}: entry is not a JSON object", i

        stored_index = entry.get("entry_index")
        if stored_index != i:
            return False, (
                f"TAMPER DETECTED at line {i}: non-contiguous entry_index "
                f"(expected {i}, got {stored_index})"
            ), stored_index if isinstance(stored_index, int) else i

        if "payload" not in entry:
            return False, f"TAMPER DETECTED at entry_index {i}: missing 'payload' field", i

        payload = entry["payload"]
        stored_prev_hash = entry.get("prev_hash")
        stored_entry_hash = entry.get("entry_hash")

        # 1. Verify prev_hash linkage
        if i == 0:
            if stored_prev_hash != GENESIS_PREV_HASH:
                return False, (
                    f"TAMPER DETECTED at entry_index 0: genesis prev_hash must be '{GENESIS_PREV_HASH}', "
                    f"got '{stored_prev_hash}'"
                ), 0
        else:
            if stored_prev_hash != prev_expected_hash:
                return False, (
                    f"TAMPER DETECTED at entry_index {i}: prev_hash mismatch "
                    f"(stored '{stored_prev_hash}', expected '{prev_expected_hash}')"
                ), i

        # 2. Recompute and verify entry_hash
        recomputed_entry_hash = compute_entry_hash(i, payload, stored_prev_hash)
        if stored_entry_hash != recomputed_entry_hash:
            return False, (
                f"TAMPER DETECTED at entry_index {i}: entry_hash mismatch "
                f"(stored '{stored_entry_hash}', reconstructed '{recomputed_entry_hash}')"
            ), i

        prev_expected_hash = stored_entry_hash

    return True, f"OK: {len(lines)} entries, chain intact", None


def main() -> int:
    log_path = sys.argv[1] if len(sys.argv) > 1 else "audit_log.jsonl"
    is_valid, msg, _ = verify_audit_log(log_path)
    if is_valid:
        print(msg)
        return 0
    else:
        print(msg, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
