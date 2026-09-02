"""
threshold_config.py — single source of truth for the operating threshold.

Every pipeline file that needs CHOSEN_THRESHOLD imports it from here.
No file hardcodes the value itself anymore.

How the threshold is resolved (in priority order):
  1. pooled_threshold_selection_summary.json exists in the same directory
     as this file → use recommended_threshold from that JSON.
     This file is written by pooled_threshold_selection.py. Running that
     script once is all that's needed to update the threshold everywhere.
  2. Fallback hardcoded value below → used only if the JSON does not exist
     yet (e.g. on a fresh clone before pooled_threshold_selection.py has
     been run). A warning is printed to stderr so the situation is visible.

To update the threshold without touching any code:
    python pooled_threshold_selection.py
That's it. All downstream files (final_threshold_report.py,
bootstrap_threshold_ci.py, risk_scoring.py, app_interactive.py) will pick
up the new value automatically on their next run.
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fallback — used only when pooled_threshold_selection_summary.json is absent.
# This was the threshold chosen in Day 4 Phase 3 from the single canonical
# dataset; it is intentionally left here as the safe default rather than
# being deleted, so this module always produces a usable value.
# ---------------------------------------------------------------------------
_FALLBACK_THRESHOLD = 0.48111024428768

_SUMMARY_FILE = Path(__file__).resolve().parent / "pooled_threshold_selection_summary.json"


def _load_threshold() -> float:
    if _SUMMARY_FILE.exists():
        try:
            data = json.loads(_SUMMARY_FILE.read_text(encoding="utf-8"))
            t = float(data["recommended_threshold"])
            return t
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"[threshold_config] WARNING: could not parse "
                f"pooled_threshold_selection_summary.json ({exc}); "
                f"falling back to hardcoded default {_FALLBACK_THRESHOLD}.",
                file=sys.stderr,
            )
            return _FALLBACK_THRESHOLD
    else:
        print(
            f"[threshold_config] NOTE: pooled_threshold_selection_summary.json "
            f"not found. Using fallback threshold {_FALLBACK_THRESHOLD}. "
            f"Run pooled_threshold_selection.py to generate a data-driven value.",
            file=sys.stderr,
        )
        return _FALLBACK_THRESHOLD


# Module-level constant — evaluated once at import time.
CHOSEN_THRESHOLD: float = _load_threshold()
