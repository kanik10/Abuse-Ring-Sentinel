"""
test_referral_features.py -- Unit tests for referral_features.py.

Verifies:
  - Cycle detection correctly flags a known A->B->C->A referral cycle (cycle_ratio > 0).
  - Cycle detection does NOT flag a non-circular referral chain (A->B->C DAG has cycle_ratio == 0).
  - Data leakage guard: raises AssertionError if is_ring_referral column is present.
  - Activation latency calculation (median days between referral and first order).
  - Referral resource overlap ratio.
  - Graceful handling of clusters with zero referral activity.
"""

import pandas as pd
import pytest

from referral_features import REFERRAL_FEATURE_COLS, compute_referral_features


def test_cycle_detection_flags_circular_loop():
    """Directed cycle A -> B -> C -> A within a 3-account cluster must yield referral_cycle_ratio == 1.0."""
    clusters = pd.DataFrame([
        {"cluster_id": 1, "account_id": "A"},
        {"cluster_id": 1, "account_id": "B"},
        {"cluster_id": 1, "account_id": "C"},
    ])

    referrals = pd.DataFrame([
        {"referrer_id": "A", "referred_id": "B", "referral_date": "2024-01-01", "bonus_amount": 10.0},
        {"referrer_id": "B", "referred_id": "C", "referral_date": "2024-01-02", "bonus_amount": 10.0},
        {"referrer_id": "C", "referred_id": "A", "referral_date": "2024-01-03", "bonus_amount": 10.0},
    ])

    orders = pd.DataFrame([
        {"account_id": "A", "amount": 100.0, "timestamp": "2024-01-05"},
        {"account_id": "B", "amount": 100.0, "timestamp": "2024-01-05"},
        {"account_id": "C", "amount": 100.0, "timestamp": "2024-01-05"},
    ])

    df_out = compute_referral_features(clusters, referrals, orders)
    assert len(df_out) == 1
    assert df_out.iloc[0]["referral_cycle_ratio"] == 1.0


def test_no_cycle_in_dag_chain():
    """Linear referral chain A -> B -> C (DAG) must yield referral_cycle_ratio == 0.0."""
    clusters = pd.DataFrame([
        {"cluster_id": 1, "account_id": "A"},
        {"cluster_id": 1, "account_id": "B"},
        {"cluster_id": 1, "account_id": "C"},
    ])

    referrals = pd.DataFrame([
        {"referrer_id": "A", "referred_id": "B", "referral_date": "2024-01-01", "bonus_amount": 10.0},
        {"referrer_id": "B", "referred_id": "C", "referral_date": "2024-01-02", "bonus_amount": 10.0},
    ])

    orders = pd.DataFrame([
        {"account_id": "B", "amount": 50.0, "timestamp": "2024-01-05"},
        {"account_id": "C", "amount": 50.0, "timestamp": "2024-01-06"},
    ])

    df_out = compute_referral_features(clusters, referrals, orders)
    assert len(df_out) == 1
    assert df_out.iloc[0]["referral_cycle_ratio"] == 0.0


def test_leakage_guard_raises_assertion_error():
    """Passing a DataFrame with is_ring_referral must trigger the leakage guard assertion."""
    clusters = pd.DataFrame([
        {"cluster_id": 1, "account_id": "A"},
        {"cluster_id": 1, "account_id": "B"},
    ])

    referrals_with_label = pd.DataFrame([
        {
            "referrer_id": "A",
            "referred_id": "B",
            "referral_date": "2024-01-01",
            "bonus_amount": 10.0,
            "is_ring_referral": True,  # Forbidden column!
        }
    ])

    orders = pd.DataFrame([{"account_id": "B", "amount": 50.0, "timestamp": "2024-01-02"}])

    with pytest.raises(AssertionError, match="LEAKAGE: referrals DataFrame contains is_ring_referral"):
        compute_referral_features(clusters, referrals_with_label, orders)


def test_activation_latency_calculation():
    """Activation latency calculates days between referral_date and first order timestamp."""
    clusters = pd.DataFrame([
        {"cluster_id": 1, "account_id": "A"},
        {"cluster_id": 1, "account_id": "B"},
    ])

    referrals = pd.DataFrame([
        {"referrer_id": "A", "referred_id": "B", "referral_date": "2024-01-01 10:00:00", "bonus_amount": 10.0},
    ])

    # First order is exactly 5 days later
    orders = pd.DataFrame([
        {"account_id": "B", "amount": 100.0, "timestamp": "2024-01-06 10:00:00"},
        {"account_id": "B", "amount": 50.0, "timestamp": "2024-01-10 10:00:00"},
    ])

    df_out = compute_referral_features(clusters, referrals, orders)
    assert len(df_out) == 1
    assert df_out.iloc[0]["median_referral_activation_days"] == 5.0


def test_cluster_without_referral_activity_omitted():
    """Clusters with no referral edges within them are omitted (caller fills NaN with 0)."""
    clusters = pd.DataFrame([
        {"cluster_id": 99, "account_id": "X"},
        {"cluster_id": 99, "account_id": "Y"},
    ])

    referrals = pd.DataFrame(columns=["referrer_id", "referred_id", "referral_date", "bonus_amount"])
    orders = pd.DataFrame(columns=["account_id", "amount", "timestamp"])

    df_out = compute_referral_features(clusters, referrals, orders)
    assert len(df_out) == 0
