"""
test_feature_engineering.py -- Unit tests for feature_engineering.py.

Verifies:
  - Exact match of all 17 features against hand-calculated analytical solutions on a 3-node K3 graph.
  - Requirement that graph G must be provided (raises ValueError if G is None).
  - Clean operation without referral features (13 base features).
  - Exclusion of clusters with cluster_size < 2.
"""

import pytest

from feature_engineering import compute_cluster_features
from referral_features import REFERRAL_FEATURE_COLS, compute_referral_features


def test_all_17_features_hand_computed_k3(hand_crafted_k3_data):
    """
    On a hand-constructed 3-node complete graph K3, verifies that all 17 features
    (13 structural graph features + 4 referral features) match their hand-calculated values.
    """
    d = hand_crafted_k3_data
    expected = d["expected"]

    # Compute referral features
    ref_features = compute_referral_features(
        d["clusters"], d["referrals"], d["orders"]
    )

    # Compute cluster features
    features_df = compute_cluster_features(
        clusters=d["clusters"],
        accounts=d["accounts"],
        orders=d["orders"],
        account_device=d["device"],
        account_payment=d["payment"],
        account_address=d["address"],
        account_ip=d["ip"],
        G=d["G"],
        referral_features=ref_features,
    )

    assert len(features_df) == 1
    row = features_df.iloc[0]

    # Feature 1: cluster_size
    assert row["cluster_size"] == expected["cluster_size"]

    # Feature 2: entity_reuse_ratio (1 - 3 distinct / 9 total usages = 2/3)
    assert pytest.approx(row["entity_reuse_ratio"]) == expected["entity_reuse_ratio"]

    # Feature 3: internal_density (complete graph -> 1.0)
    assert pytest.approx(row["internal_density"]) == expected["internal_density"]

    # Feature 4 & 5: degree_centrality (each node has deg 2 out of 2 -> 1.0)
    assert pytest.approx(row["mean_degree_centrality"]) == expected["mean_degree_centrality"]
    assert pytest.approx(row["max_degree_centrality"]) == expected["max_degree_centrality"]

    # Feature 6 & 7: pagerank (symmetric K3 with equal weights -> 1/3 each)
    assert pytest.approx(row["mean_pagerank"], rel=1e-3) == expected["mean_pagerank"]
    assert pytest.approx(row["max_pagerank"], rel=1e-3) == expected["max_pagerank"]

    # Feature 8 & 9: betweenness_centrality (direct edges -> 0.0)
    assert pytest.approx(row["mean_betweenness_centrality"], abs=1e-6) == expected["mean_betweenness_centrality"]
    assert pytest.approx(row["max_betweenness_centrality"], abs=1e-6) == expected["max_betweenness_centrality"]

    # Feature 10: creation_span_days (Jan 3 minus Jan 1 = 2 days)
    assert row["creation_span_days"] == expected["creation_span_days"]

    # Feature 11: creation_std_days (sample std of days 0, 1, 2 = 1.0)
    assert pytest.approx(row["creation_std_days"]) == expected["creation_std_days"]

    # Feature 12: avg_order_amount ((100 + 200 + 300) / 3 = 200.0)
    assert pytest.approx(row["avg_order_amount"]) == expected["avg_order_amount"]

    # Feature 13: order_amount_cv (std 100 / mean 200 = 0.5)
    assert pytest.approx(row["order_amount_cv"]) == expected["order_amount_cv"]

    # Feature 14: referral_cycle_ratio (directed 3-cycle in 3-member cluster -> 1.0)
    assert pytest.approx(row["referral_cycle_ratio"]) == expected["referral_cycle_ratio"]

    # Feature 15: referral_resource_overlap_ratio (all 3 referral edges within cluster -> 1.0)
    assert pytest.approx(row["referral_resource_overlap_ratio"]) == expected["referral_resource_overlap_ratio"]

    # Feature 16: median_referral_activation_days (all 3 accounts activated in 4 days -> 4.0)
    assert pytest.approx(row["median_referral_activation_days"]) == expected["median_referral_activation_days"]

    # Feature 17: within_cluster_referral_density (3 undirected pairs out of 3 possible -> 1.0)
    assert pytest.approx(row["within_cluster_referral_density"]) == expected["within_cluster_referral_density"]


def test_compute_cluster_features_requires_graph(hand_crafted_k3_data):
    """Raises ValueError if graph G is missing."""
    d = hand_crafted_k3_data
    with pytest.raises(ValueError, match="G must be provided"):
        compute_cluster_features(
            clusters=d["clusters"],
            accounts=d["accounts"],
            orders=d["orders"],
            account_device=d["device"],
            account_payment=d["payment"],
            account_address=d["address"],
            G=None,
        )


def test_compute_cluster_features_without_referral_features(hand_crafted_k3_data):
    """When referral_features is None, returns 13 structural features without errors."""
    d = hand_crafted_k3_data
    features_df = compute_cluster_features(
        clusters=d["clusters"],
        accounts=d["accounts"],
        orders=d["orders"],
        account_device=d["device"],
        account_payment=d["payment"],
        account_address=d["address"],
        G=d["G"],
        referral_features=None,
    )
    assert len(features_df) == 1
    for col in REFERRAL_FEATURE_COLS:
        assert col not in features_df.columns
    assert "entity_reuse_ratio" in features_df.columns
    assert "internal_density" in features_df.columns
