"""
test_audit_chain.py -- Unit and integrity tests for audit_chain.py and verify_audit_log.py.

Verifies:
  - Determinism: repeated build_chain() and write_chain() calls produce 100% byte-identical files.
  - Genesis entry: entry_index 0 has prev_hash == "0" * 64.
  - Chain linkage: entry N's prev_hash matches entry N-1's entry_hash.
  - Tamper detection: mutating a field inside entry_index 1's payload localizes the break to entry_index 1.
  - Empty case: 0 entries produces empty file and verifier reports "OK: 0 entries, chain intact".
  - Index discontinuity: non-contiguous entry_index sequence is detected and flagged.
"""

from pathlib import Path
import pytest

from audit_chain import GENESIS_PREV_HASH, build_chain, canonical_json, sha256_hex, write_chain
from verify_audit_log import verify_audit_log


@pytest.fixture
def sample_payloads():
    return [
        {
            "cluster_id": 1,
            "risk_score": 0.8542,
            "member_account_ids": ["ACC_A", "ACC_B"],
            "shared_resources": ["device:DEV_01 (used by 2 members)"],
            "contributing_features": {"cluster_size": 2, "entity_reuse_ratio": 0.5},
            "account_scores": [
                {"account_id": "ACC_A", "account_risk_score": 0.9},
                {"account_id": "ACC_B", "account_risk_score": 0.7},
            ],
            "recommended_action": "flag_for_review",
        },
        {
            "cluster_id": 2,
            "risk_score": 0.7123,
            "member_account_ids": ["ACC_C", "ACC_D"],
            "shared_resources": ["payment:PAY_01 (used by 2 members)"],
            "contributing_features": {"cluster_size": 2, "entity_reuse_ratio": 0.5},
            "account_scores": [
                {"account_id": "ACC_C", "account_risk_score": 0.8},
                {"account_id": "ACC_D", "account_risk_score": 0.6},
            ],
            "recommended_action": "flag_for_review",
        },
        {
            "cluster_id": 3,
            "risk_score": 0.9912,
            "member_account_ids": ["ACC_E", "ACC_F", "ACC_G"],
            "shared_resources": ["ip:IP_01 (used by 3 members)"],
            "contributing_features": {"cluster_size": 3, "entity_reuse_ratio": 0.6667},
            "account_scores": [],
            "recommended_action": "flag_for_review",
        },
    ]


def test_determinism_byte_identical(sample_payloads, tmp_path):
    """
    Building and writing the same payloads twice produces byte-identical files.
    Raw bytes must match exactly (no wall-clock timestamps or nondeterministic keys).
    """
    file1 = tmp_path / "audit_run1.jsonl"
    file2 = tmp_path / "audit_run2.jsonl"

    chain1 = build_chain(sample_payloads)
    chain2 = build_chain(sample_payloads)

    write_chain(chain1, file1)
    write_chain(chain2, file2)

    bytes1 = file1.read_bytes()
    bytes2 = file2.read_bytes()

    assert bytes1 == bytes2
    assert len(bytes1) > 0


def test_genesis_entry_prev_hash(sample_payloads):
    """Genesis entry (index 0) must have prev_hash == '0' * 64."""
    chain = build_chain(sample_payloads)
    assert len(chain) == 3
    assert chain[0]["entry_index"] == 0
    assert chain[0]["prev_hash"] == GENESIS_PREV_HASH
    assert chain[0]["prev_hash"] == "0" * 64


def test_chain_linkage(sample_payloads):
    """
    Each entry N's prev_hash must strictly match entry N-1's entry_hash,
    which is the SHA-256 digest of entry N-1's canonical JSON content.
    """
    chain = build_chain(sample_payloads)
    for i in range(1, len(chain)):
        prev_entry = chain[i - 1]
        curr_entry = chain[i]

        assert curr_entry["prev_hash"] == prev_entry["entry_hash"]

        # Confirm entry_hash matches recomputed SHA-256 of entry body
        expected_body = {
            "entry_index": prev_entry["entry_index"],
            "payload": prev_entry["payload"],
            "prev_hash": prev_entry["prev_hash"],
        }
        assert prev_entry["entry_hash"] == sha256_hex(canonical_json(expected_body))


def test_tamper_detection_localizes_break(sample_payloads, tmp_path):
    """
    Mutating a single field inside entry_index 1's payload breaks the chain
    and verify_audit_log accurately localizes the break to entry_index 1.
    """
    chain = build_chain(sample_payloads)
    audit_file = tmp_path / "tampered_audit.jsonl"
    write_chain(chain, audit_file)

    # Valid chain passes verification
    is_valid, msg, broken_idx = verify_audit_log(audit_file)
    assert is_valid is True
    assert broken_idx is None
    assert "OK: 3 entries, chain intact" in msg

    # Tamper with entry_index 1's risk_score
    chain[1]["payload"]["risk_score"] = 0.9999
    write_chain(chain, audit_file)

    # Verifier must catch tamper and report entry_index 1
    is_valid, msg, broken_idx = verify_audit_log(audit_file)
    assert is_valid is False
    assert broken_idx == 1
    assert "TAMPER DETECTED at entry_index 1" in msg
    assert "entry_hash mismatch" in msg


def test_empty_case_handling(tmp_path):
    """Zero flagged clusters produces an empty file, which verifies successfully."""
    empty_file = tmp_path / "empty_audit.jsonl"
    chain = build_chain([])
    write_chain(chain, empty_file)

    assert empty_file.read_text(encoding="utf-8") == ""

    is_valid, msg, broken_idx = verify_audit_log(empty_file)
    assert is_valid is True
    assert broken_idx is None
    assert "OK: 0 entries, chain intact" in msg


def test_index_discontinuity_detection(sample_payloads, tmp_path):
    """Skipping an entry_index (e.g. 0 -> 2) is flagged as a tamper event."""
    chain = build_chain(sample_payloads)
    # Tamper index sequence: make entry 1 have index 5
    chain[1]["entry_index"] = 5
    audit_file = tmp_path / "discontinuous.jsonl"
    write_chain(chain, audit_file)

    is_valid, msg, broken_idx = verify_audit_log(audit_file)
    assert is_valid is False
    assert broken_idx == 5
    assert "non-contiguous entry_index" in msg
