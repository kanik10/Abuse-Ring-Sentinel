"""
test_threshold_config.py -- Unit and regression tests for threshold_config.py.

Verifies:
  - Regression test: CHOSEN_THRESHOLD in threshold_config matches what
    final_threshold_report.py and risk_scoring.py actually import (single source of truth).
  - The resolved operating threshold value equals 0.1333192760434377.
  - _load_threshold falls back to _FALLBACK_THRESHOLD when the summary JSON is missing.
  - _load_threshold falls back to _FALLBACK_THRESHOLD when the summary JSON is corrupt or invalid.
  - _load_threshold correctly parses valid custom JSON files.
"""

import json
from pathlib import Path
import pytest

import final_threshold_report
import risk_scoring
import threshold_config
from threshold_config import _FALLBACK_THRESHOLD, _load_threshold


def test_chosen_threshold_regression_single_source():
    """
    Regression test: CHOSEN_THRESHOLD in threshold_config MUST match what
    final_threshold_report.py and risk_scoring.py actually import.
    Guards against duplication bugs where modules hardcode divergent thresholds.
    """
    assert threshold_config.CHOSEN_THRESHOLD == final_threshold_report.CHOSEN_THRESHOLD
    assert threshold_config.CHOSEN_THRESHOLD == risk_scoring.CHOSEN_THRESHOLD


def test_chosen_threshold_value():
    """The canonical locked operating threshold is 0.1333192760434377."""
    assert pytest.approx(threshold_config.CHOSEN_THRESHOLD, abs=1e-12) == 0.1333192760434377


def test_fallback_when_summary_file_missing(monkeypatch, tmp_path):
    """When pooled_threshold_selection_summary.json is absent, falls back to _FALLBACK_THRESHOLD."""
    non_existent = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(threshold_config, "_SUMMARY_FILE", non_existent)

    val = _load_threshold()
    assert val == _FALLBACK_THRESHOLD


def test_fallback_when_summary_file_corrupt(monkeypatch, tmp_path):
    """When summary JSON is malformed, falls back to _FALLBACK_THRESHOLD."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid_json: true", encoding="utf-8")
    monkeypatch.setattr(threshold_config, "_SUMMARY_FILE", corrupt_file)

    val = _load_threshold()
    assert val == _FALLBACK_THRESHOLD


def test_fallback_when_summary_file_missing_key(monkeypatch, tmp_path):
    """When summary JSON lacks recommended_threshold key, falls back to _FALLBACK_THRESHOLD."""
    incomplete_file = tmp_path / "incomplete.json"
    incomplete_file.write_text(json.dumps({"other_key": 0.5}), encoding="utf-8")
    monkeypatch.setattr(threshold_config, "_SUMMARY_FILE", incomplete_file)

    val = _load_threshold()
    assert val == _FALLBACK_THRESHOLD


def test_load_valid_summary_json(monkeypatch, tmp_path):
    """When summary JSON contains valid recommended_threshold, loads it correctly."""
    valid_file = tmp_path / "valid.json"
    valid_file.write_text(json.dumps({"recommended_threshold": 0.25}), encoding="utf-8")
    monkeypatch.setattr(threshold_config, "_SUMMARY_FILE", valid_file)

    val = _load_threshold()
    assert pytest.approx(val) == 0.25
