"""
test_entity_resolution.py -- Unit tests for entity_resolution.py.

Verifies:
  - Correct entity mapping/normalization via sorted-neighborhood string similarity.
  - No orphaned IDs (all observed strings are preserved in the mapping dictionary).
  - resolve_mapping_table correctly updates DataFrame columns.
  - Edge cases: empty lists, single IDs, length difference > 1, identical duplicates.
"""

import pandas as pd
import pytest

from entity_resolution import resolve, resolve_mapping_table


def test_resolve_similar_strings_unified():
    """Near-identical observed strings (e.g. 1 char perturbation) must resolve to the same root."""
    observed = [
        "DEV_IPHONE_12_PRO_ABC1",
        "DEV_IPHONE_12_PRO_ABC2",  # 1 char diff, > 0.85 ratio
        "DEV_SAMSUNG_GALAXY_999",  # completely different
    ]
    mapping = resolve(observed, window=8)

    # Similar strings resolve to the same root
    assert mapping["DEV_IPHONE_12_PRO_ABC1"] == mapping["DEV_IPHONE_12_PRO_ABC2"]
    # Dissimilar string stays separate
    assert mapping["DEV_SAMSUNG_GALAXY_999"] != mapping["DEV_IPHONE_12_PRO_ABC1"]


def test_resolve_dissimilar_strings_distinct():
    """Dissimilar strings must remain distinct entities."""
    observed = ["DEV_A100", "DEV_B200", "DEV_C300"]
    mapping = resolve(observed, window=8)
    assert len(set(mapping.values())) == 3
    for s in observed:
        assert mapping[s] == s


def test_resolve_no_orphaned_ids():
    """Every observed ID must be present in the returned mapping (no orphans)."""
    observed = [
        "ADDR_100_MAIN_ST_APT_1A",
        "ADDR_100_MAIN_ST_APT_1B",
        "ADDR_500_FIFTH_AVE",
        "ADDR_999_BROADWAY_SUITE_10",
    ]
    mapping = resolve(observed, window=8)

    # No missing keys
    assert set(mapping.keys()) == set(observed)

    # Every mapped value must be a valid representative from the input set
    for orig, rep in mapping.items():
        assert rep in observed


def test_resolve_length_diff_exceeds_one():
    """Strings differing in length by > 1 char must not be unified by sorted-neighborhood."""
    observed = ["DEV_TEST_1", "DEV_TEST_100"]
    mapping = resolve(observed, window=8)
    assert mapping["DEV_TEST_1"] != mapping["DEV_TEST_100"]


def test_resolve_empty_and_single():
    """Edge cases: empty list and single string."""
    assert resolve([]) == {}
    assert resolve(["DEV_ONLY_ONE"]) == {"DEV_ONLY_ONE": "DEV_ONLY_ONE"}


def test_resolve_duplicates_in_input():
    """Duplicate observed IDs are handled gracefully and map to the same representative."""
    observed = ["DEV_DUPE_01", "DEV_DUPE_01", "DEV_DUPE_02"]
    mapping = resolve(observed, window=8)
    assert "DEV_DUPE_01" in mapping
    assert "DEV_DUPE_02" in mapping


def test_resolve_mapping_table():
    """resolve_mapping_table must map the target column without dropping rows or introducing NaNs."""
    df = pd.DataFrame({
        "account_id": ["ACC_1", "ACC_2", "ACC_3"],
        "device_id": [
            "DEV_PIXEL_7_VARIANT_A",
            "DEV_PIXEL_7_VARIANT_B",
            "DEV_IPHONE_UNIQUE_X",
        ]
    })

    resolved_df = resolve_mapping_table(df, "device_id")

    assert len(resolved_df) == 3
    assert resolved_df["device_id"].notna().all()
    # The two variants should now share the same device_id
    assert resolved_df.loc[0, "device_id"] == resolved_df.loc[1, "device_id"]
    # The unique one remains distinct
    assert resolved_df.loc[2, "device_id"] != resolved_df.loc[0, "device_id"]
