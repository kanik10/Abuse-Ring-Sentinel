"""
test_naive_baseline.py -- Unit tests for naive_baseline.py.

Verifies:
  - parse_bool_labels properly handles boolean and case-insensitive string values.
  - parse_bool_labels raises ValueError on unrecognized non-boolean strings.
  - require_columns enforces required schema columns and raises informative ValueError.
  - build_flagged_accounts_by_threshold groups and flags accounts sharing addresses.
"""

from pathlib import Path
import pandas as pd
import pytest

from naive_baseline import (
    build_flagged_accounts_by_threshold,
    parse_bool_labels,
    require_columns,
)


def test_parse_bool_labels_boolean_input():
    """Native boolean series passes through unmodified."""
    s = pd.Series([True, False, True])
    parsed = parse_bool_labels(s)
    assert parsed.equals(s)


def test_parse_bool_labels_string_input():
    """String representations like 'true', 'FALSE', ' True ' parse to booleans."""
    s = pd.Series(["true", "FALSE", " True ", "false"])
    parsed = parse_bool_labels(s)
    assert list(parsed) == [True, False, True, False]


def test_parse_bool_labels_invalid_strings():
    """Non-boolean string values raise ValueError with bad values listed."""
    s = pd.Series(["true", "unknown", "maybe"])
    with pytest.raises(ValueError, match="contains non-boolean values"):
        parse_bool_labels(s)


def test_require_columns_passes():
    """require_columns passes silently when all required columns are present."""
    df = pd.DataFrame({"col_a": [1], "col_b": [2]})
    require_columns(df, ["col_a", "col_b"], Path("test.csv"))


def test_require_columns_raises_on_missing():
    """require_columns raises ValueError when required column is absent."""
    df = pd.DataFrame({"col_a": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(df, ["col_a", "missing_col"], Path("test.csv"))


def test_build_flagged_accounts_by_threshold():
    """Accounts sharing an address with >= threshold unique accounts are flagged."""
    # ADDR_SHARED used by 4 accounts
    address_df = pd.DataFrame([
        {"account_id": f"ACC_{i}", "address_id": "ADDR_SHARED"} for i in range(4)
    ])

    flagged_dict = build_flagged_accounts_by_threshold(address_df)

    # Threshold 3 and 4: all 4 accounts should be flagged
    assert len(flagged_dict[3]) == 4
    assert len(flagged_dict[4]) == 4
    # Threshold 5: 0 accounts flagged (degree is 4 < 5)
    assert len(flagged_dict[5]) == 0
