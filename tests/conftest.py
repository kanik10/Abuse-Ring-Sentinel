"""
conftest.py -- Deterministic synthetic test fixtures for Abuse-Ring Sentinel.

Provides lightweight, isolated, in-memory fixtures (DataFrames and NetworkX graphs)
representing:
  1. A fraud ring cluster (tight resource sharing, directed referral cycle, synchronized burst).
  2. A benign coincidental cluster (shared apartment address only, distinct devices/payments/IPs, no referral cycle).
  3. An isolated account (unique resources, no sharing).
  4. A hub vs. peripheral cluster (for intra-cluster account ranking).
  5. A hand-computable 3-node complete graph (K3) fixture with exact analytical solutions for all 17 features.

None of these fixtures touch or depend on day1_data/ CSV files.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic Dataset 1: Fraud Ring vs Benign Coincidental + Isolated
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_accounts_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Cluster 1: Fraud Ring (burst creation around Jan 1 2024)
        {"account_id": "ACC_RING_01", "creation_date": "2024-01-01 10:00:00"},
        {"account_id": "ACC_RING_02", "creation_date": "2024-01-01 11:30:00"},
        {"account_id": "ACC_RING_03", "creation_date": "2024-01-01 12:45:00"},

        # Cluster 2: Benign Coincidental (widely spaced organic creation)
        {"account_id": "ACC_BENIGN_01", "creation_date": "2022-03-15 08:00:00"},
        {"account_id": "ACC_BENIGN_02", "creation_date": "2023-07-20 14:15:00"},
        {"account_id": "ACC_BENIGN_03", "creation_date": "2024-02-10 19:30:00"},

        # Isolated account
        {"account_id": "ACC_SOLO_01", "creation_date": "2023-11-01 12:00:00"},
    ])


@pytest.fixture
def synthetic_device_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Ring shares single device
        {"account_id": "ACC_RING_01", "device_id": "DEV_RING_SHARED"},
        {"account_id": "ACC_RING_02", "device_id": "DEV_RING_SHARED"},
        {"account_id": "ACC_RING_03", "device_id": "DEV_RING_SHARED"},

        # Benign accounts have distinct devices
        {"account_id": "ACC_BENIGN_01", "device_id": "DEV_BENIGN_01"},
        {"account_id": "ACC_BENIGN_02", "device_id": "DEV_BENIGN_02"},
        {"account_id": "ACC_BENIGN_03", "device_id": "DEV_BENIGN_03"},

        # Solo account
        {"account_id": "ACC_SOLO_01", "device_id": "DEV_SOLO_01"},
    ])


@pytest.fixture
def synthetic_payment_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Ring shares single payment token
        {"account_id": "ACC_RING_01", "payment_id": "PAY_RING_SHARED"},
        {"account_id": "ACC_RING_02", "payment_id": "PAY_RING_SHARED"},
        {"account_id": "ACC_RING_03", "payment_id": "PAY_RING_SHARED"},

        # Benign accounts have distinct payment methods
        {"account_id": "ACC_BENIGN_01", "payment_id": "PAY_BENIGN_01"},
        {"account_id": "ACC_BENIGN_02", "payment_id": "PAY_BENIGN_02"},
        {"account_id": "ACC_BENIGN_03", "payment_id": "PAY_BENIGN_03"},

        # Solo account
        {"account_id": "ACC_SOLO_01", "payment_id": "PAY_SOLO_01"},
    ])


@pytest.fixture
def synthetic_address_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Ring shares address
        {"account_id": "ACC_RING_01", "address_id": "ADDR_RING_SHARED"},
        {"account_id": "ACC_RING_02", "address_id": "ADDR_RING_SHARED"},
        {"account_id": "ACC_RING_03", "address_id": "ADDR_RING_SHARED"},

        # Benign accounts share ONLY apartment complex address
        {"account_id": "ACC_BENIGN_01", "address_id": "ADDR_APT_COMPLEX_99"},
        {"account_id": "ACC_BENIGN_02", "address_id": "ADDR_APT_COMPLEX_99"},
        {"account_id": "ACC_BENIGN_03", "address_id": "ADDR_APT_COMPLEX_99"},

        # Solo account
        {"account_id": "ACC_SOLO_01", "address_id": "ADDR_SOLO_01"},
    ])


@pytest.fixture
def synthetic_ip_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Ring shares IP
        {"account_id": "ACC_RING_01", "ip_id": "IP_RING_SHARED"},
        {"account_id": "ACC_RING_02", "ip_id": "IP_RING_SHARED"},
        {"account_id": "ACC_RING_03", "ip_id": "IP_RING_SHARED"},

        # Benign accounts have distinct IPs
        {"account_id": "ACC_BENIGN_01", "ip_id": "IP_BENIGN_01"},
        {"account_id": "ACC_BENIGN_02", "ip_id": "IP_BENIGN_02"},
        {"account_id": "ACC_BENIGN_03", "ip_id": "IP_BENIGN_03"},

        # Solo account
        {"account_id": "ACC_SOLO_01", "ip_id": "IP_SOLO_01"},
    ])


@pytest.fixture
def synthetic_orders_df() -> pd.DataFrame:
    return pd.DataFrame([
        # Ring orders
        {"account_id": "ACC_RING_01", "amount": 150.0, "timestamp": "2024-01-02 10:00:00"},
        {"account_id": "ACC_RING_02", "amount": 150.0, "timestamp": "2024-01-02 11:00:00"},
        {"account_id": "ACC_RING_03", "amount": 150.0, "timestamp": "2024-01-02 12:00:00"},

        # Benign orders
        {"account_id": "ACC_BENIGN_01", "amount": 35.0, "timestamp": "2022-04-01 15:00:00"},
        {"account_id": "ACC_BENIGN_02", "amount": 250.0, "timestamp": "2023-08-10 18:00:00"},
        {"account_id": "ACC_BENIGN_03", "amount": 90.0, "timestamp": "2024-03-01 12:00:00"},

        # Solo order
        {"account_id": "ACC_SOLO_01", "amount": 45.0, "timestamp": "2023-11-05 09:00:00"},
    ])


@pytest.fixture
def synthetic_referrals_df() -> pd.DataFrame:
    """Referrals without is_ring_referral column (leakage guard compliant)."""
    return pd.DataFrame([
        # Ring 3-cycle: 1 -> 2 -> 3 -> 1
        {"referrer_id": "ACC_RING_01", "referred_id": "ACC_RING_02", "referral_date": "2024-01-01 11:00:00", "bonus_amount": 50.0},
        {"referrer_id": "ACC_RING_02", "referred_id": "ACC_RING_03", "referral_date": "2024-01-01 12:00:00", "bonus_amount": 50.0},
        {"referrer_id": "ACC_RING_03", "referred_id": "ACC_RING_01", "referral_date": "2024-01-01 13:00:00", "bonus_amount": 50.0},

        # Benign organic referral: linear DAG 1 -> 2
        {"referrer_id": "ACC_BENIGN_01", "referred_id": "ACC_BENIGN_02", "referral_date": "2023-07-20 10:00:00", "bonus_amount": 20.0},
    ])


@pytest.fixture
def synthetic_clusters_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"account_id": "ACC_RING_01", "cluster_id": 1, "cluster_size": 3},
        {"account_id": "ACC_RING_02", "cluster_id": 1, "cluster_size": 3},
        {"account_id": "ACC_RING_03", "cluster_id": 1, "cluster_size": 3},
        {"account_id": "ACC_BENIGN_01", "cluster_id": 2, "cluster_size": 3},
        {"account_id": "ACC_BENIGN_02", "cluster_id": 2, "cluster_size": 3},
        {"account_id": "ACC_BENIGN_03", "cluster_id": 2, "cluster_size": 3},
    ])


@pytest.fixture
def synthetic_ground_truth_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"account_id": "ACC_RING_01", "is_ring_member": True, "ring_id": "RING_01", "coincidental_group_id": np.nan},
        {"account_id": "ACC_RING_02", "is_ring_member": True, "ring_id": "RING_01", "coincidental_group_id": np.nan},
        {"account_id": "ACC_RING_03", "is_ring_member": True, "ring_id": "RING_01", "coincidental_group_id": np.nan},
        {"account_id": "ACC_BENIGN_01", "is_ring_member": False, "ring_id": np.nan, "coincidental_group_id": "BENIGN_GRP_01"},
        {"account_id": "ACC_BENIGN_02", "is_ring_member": False, "ring_id": np.nan, "coincidental_group_id": "BENIGN_GRP_01"},
        {"account_id": "ACC_BENIGN_03", "is_ring_member": False, "ring_id": np.nan, "coincidental_group_id": "BENIGN_GRP_01"},
        {"account_id": "ACC_SOLO_01", "is_ring_member": False, "ring_id": np.nan, "coincidental_group_id": np.nan},
    ])


# ---------------------------------------------------------------------------
# Hand-Constructed K3 Fixture (Exact Analytical Solutions for All 17 Features)
# ---------------------------------------------------------------------------

@pytest.fixture
def hand_crafted_k3_data():
    """
    Constructs an exact 3-node complete graph K3 cluster (cluster_id=10)
    where all 17 features can be hand-calculated cleanly.

    Nodes: ACC_K3_A, ACC_K3_B, ACC_K3_C
    Resources:
      - Device: DEV_K3_ALL (shared by all 3 -> degree 3 -> weight 1/3)
      - Payment: PAY_K3_ALL (shared by all 3 -> degree 3 -> weight 1/3)
      - Address: ADDR_K3_ALL (shared by all 3 -> degree 3 -> weight 1/3)
      - IP: None
    Edge weight per pair = 1/3 + 1/3 + 1/3 = 1.0.

    Expected 17 Features:
      1. cluster_size = 3
      2. entity_reuse_ratio = 1 - (3 distinct / 9 total usages) = 2/3 ≈ 0.666666...
      3. internal_density = 1.0 (3 edges / 3 possible)
      4. mean_degree_centrality = 1.0
      5. max_degree_centrality = 1.0
      6. mean_pagerank = 1/3 ≈ 0.333333...
      7. max_pagerank = 1/3 ≈ 0.333333...
      8. mean_betweenness_centrality = 0.0
      9. max_betweenness_centrality = 0.0
      10. creation_span_days = 2 (2024-01-03 minus 2024-01-01)
      11. creation_std_days = 1.0 (sample std of days 0, 1, 2 is 1.0)
      12. avg_order_amount = 200.0 (mean of 100, 200, 300)
      13. order_amount_cv = 0.5 (sample std 100 / mean 200)
      14. referral_cycle_ratio = 1.0 (3 cycle nodes / 3 members)
      15. referral_resource_overlap_ratio = 1.0 (3 edges in cluster / 3 total involving members)
      16. median_referral_activation_days = 4.0 (activation delays: [4, 4, 4])
      17. within_cluster_referral_density = 1.0 (3 undirected pairs / 3 possible)
    """
    members = ["ACC_K3_A", "ACC_K3_B", "ACC_K3_C"]

    clusters = pd.DataFrame([
        {"account_id": m, "cluster_id": 10, "cluster_size": 3} for m in members
    ])

    accounts = pd.DataFrame([
        {"account_id": "ACC_K3_A", "creation_date": "2024-01-01 00:00:00"},
        {"account_id": "ACC_K3_B", "creation_date": "2024-01-02 00:00:00"},
        {"account_id": "ACC_K3_C", "creation_date": "2024-01-03 00:00:00"},
    ])

    device = pd.DataFrame([{"account_id": m, "device_id": "DEV_K3_ALL"} for m in members])
    payment = pd.DataFrame([{"account_id": m, "payment_id": "PAY_K3_ALL"} for m in members])
    address = pd.DataFrame([{"account_id": m, "address_id": "ADDR_K3_ALL"} for m in members])
    ip = None

    # Orders designed for avg=200, std=100, cv=0.5
    # and first order timestamps 4 days after respective referral dates
    orders = pd.DataFrame([
        {"account_id": "ACC_K3_A", "amount": 100.0, "timestamp": "2024-01-07 00:00:00"},
        {"account_id": "ACC_K3_B", "amount": 200.0, "timestamp": "2024-01-05 00:00:00"},
        {"account_id": "ACC_K3_C", "amount": 300.0, "timestamp": "2024-01-06 00:00:00"},
    ])

    # Directed 3-cycle: A -> B, B -> C, C -> A
    referrals = pd.DataFrame([
        {"referrer_id": "ACC_K3_A", "referred_id": "ACC_K3_B", "referral_date": "2024-01-01 00:00:00", "bonus_amount": 10.0},
        {"referrer_id": "ACC_K3_B", "referred_id": "ACC_K3_C", "referral_date": "2024-01-02 00:00:00", "bonus_amount": 10.0},
        {"referrer_id": "ACC_K3_C", "referred_id": "ACC_K3_A", "referral_date": "2024-01-03 00:00:00", "bonus_amount": 10.0},
    ])

    G = nx.Graph()
    G.add_edge("ACC_K3_A", "ACC_K3_B", weight=1.0)
    G.add_edge("ACC_K3_B", "ACC_K3_C", weight=1.0)
    G.add_edge("ACC_K3_A", "ACC_K3_C", weight=1.0)

    expected = {
        "cluster_size": 3,
        "entity_reuse_ratio": 2.0 / 3.0,
        "internal_density": 1.0,
        "mean_degree_centrality": 1.0,
        "max_degree_centrality": 1.0,
        "mean_pagerank": 1.0 / 3.0,
        "max_pagerank": 1.0 / 3.0,
        "mean_betweenness_centrality": 0.0,
        "max_betweenness_centrality": 0.0,
        "creation_span_days": 2,
        "creation_std_days": 1.0,
        "avg_order_amount": 200.0,
        "order_amount_cv": 0.5,
        "referral_cycle_ratio": 1.0,
        "referral_resource_overlap_ratio": 1.0,
        "median_referral_activation_days": 4.0,
        "within_cluster_referral_density": 1.0,
    }

    return {
        "clusters": clusters,
        "accounts": accounts,
        "orders": orders,
        "device": device,
        "payment": payment,
        "address": address,
        "ip": ip,
        "G": G,
        "referrals": referrals,
        "expected": expected,
    }


# ---------------------------------------------------------------------------
# Synthetic Hub vs Peripheral Cluster Fixture (Account Scoring)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_hub_cluster_data():
    """
    Cluster 20 has:
      - ACC_HUB: connected to 3 peripheral accounts via distinct devices and payments
      - ACC_PERIPH_1, ACC_PERIPH_2, ACC_PERIPH_3: connected ONLY to ACC_HUB
    """
    members = ["ACC_HUB", "ACC_PERIPH_1", "ACC_PERIPH_2", "ACC_PERIPH_3"]

    clusters = pd.DataFrame([
        {"account_id": m, "cluster_id": 20, "cluster_size": 4} for m in members
    ])

    device = pd.DataFrame([
        {"account_id": "ACC_HUB", "device_id": "DEV_HUB_1"},
        {"account_id": "ACC_PERIPH_1", "device_id": "DEV_HUB_1"},
        {"account_id": "ACC_HUB", "device_id": "DEV_HUB_2"},
        {"account_id": "ACC_PERIPH_2", "device_id": "DEV_HUB_2"},
        {"account_id": "ACC_HUB", "device_id": "DEV_HUB_3"},
        {"account_id": "ACC_PERIPH_3", "device_id": "DEV_HUB_3"},
    ])

    payment = pd.DataFrame([
        {"account_id": "ACC_HUB", "payment_id": "PAY_HUB_SHARED"},
        {"account_id": "ACC_PERIPH_1", "payment_id": "PAY_HUB_SHARED"},
        {"account_id": "ACC_PERIPH_2", "payment_id": "PAY_HUB_SHARED"},
        {"account_id": "ACC_PERIPH_3", "payment_id": "PAY_HUB_SHARED"},
    ])

    address = pd.DataFrame([
        {"account_id": m, "address_id": "ADDR_CLUSTER_20"} for m in members
    ])

    ip = pd.DataFrame([
        {"account_id": m, "ip_id": "IP_CLUSTER_20"} for m in members
    ])

    accounts = pd.DataFrame([
        {"account_id": m, "creation_date": "2024-01-01 12:00:00"} for m in members
    ])

    orders = pd.DataFrame([
        {"account_id": m, "amount": 100.0} for m in members
    ])

    G = nx.Graph()
    G.add_edge("ACC_HUB", "ACC_PERIPH_1", weight=2.0)
    G.add_edge("ACC_HUB", "ACC_PERIPH_2", weight=2.0)
    G.add_edge("ACC_HUB", "ACC_PERIPH_3", weight=2.0)

    return {
        "cluster_id": 20,
        "members": members,
        "clusters": clusters,
        "accounts": accounts,
        "orders": orders,
        "device": device,
        "payment": payment,
        "address": address,
        "ip": ip,
        "G": G,
    }
