"""
test_graph_builder.py -- Unit tests for graph_builder.py.

Verifies:
  - Edge weights computed as inverse resource frequency (1 / degree(resource)).
  - Accumulation of weights across multiple shared resources.
  - No self-loops exist in the constructed graph.
  - Graph is strictly undirected.
  - Isolated accounts (no shared resources) are excluded from the graph.
  - Centrality feature computation (degree, pagerank, betweenness).
  - Edge cases: empty graphs and empty inputs.
"""

import networkx as nx
import pandas as pd
import pytest

from graph_builder import (
    CENTRALITY_COLUMNS,
    build_account_graph,
    compute_account_centrality_features,
)


def test_inverse_resource_frequency_edge_weights():
    """
    Edge weight accumulates 1 / degree(resource) for every shared resource.
    - dev_shared is used by 2 accounts (A, B) -> weight += 1/2 = 0.5
    - pay_shared is used by 4 accounts (A, B, C, D) -> each pair weight += 1/4 = 0.25
    Between A and B, total weight must be 0.5 + 0.25 = 0.75.
    Between C and D, total weight must be 0.25.
    """
    device_df = pd.DataFrame([
        {"account_id": "A", "device_id": "DEV_SHARED_2"},
        {"account_id": "B", "device_id": "DEV_SHARED_2"},
    ])

    payment_df = pd.DataFrame([
        {"account_id": "A", "payment_id": "PAY_SHARED_4"},
        {"account_id": "B", "payment_id": "PAY_SHARED_4"},
        {"account_id": "C", "payment_id": "PAY_SHARED_4"},
        {"account_id": "D", "payment_id": "PAY_SHARED_4"},
    ])

    empty_addr = pd.DataFrame(columns=["account_id", "address_id"])

    G = build_account_graph(device_df, payment_df, empty_addr)

    assert G.has_edge("A", "B")
    assert pytest.approx(G["A"]["B"]["weight"]) == 0.75

    assert G.has_edge("C", "D")
    assert pytest.approx(G["C"]["D"]["weight"]) == 0.25

    assert G.has_edge("A", "C")
    assert pytest.approx(G["A"]["C"]["weight"]) == 0.25


def test_no_self_loops(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df):
    """The constructed graph must contain strictly zero self-loops."""
    G = build_account_graph(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df)
    assert len(list(nx.selfloop_edges(G))) == 0
    for node in G.nodes:
        assert not G.has_edge(node, node)


def test_graph_is_undirected(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df):
    """The graph must be an undirected NetworkX graph."""
    G = build_account_graph(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df)
    assert isinstance(G, nx.Graph)
    assert not G.is_directed()
    # Undirected property: edge (u, v) is identical to (v, u)
    for u, v in G.edges():
        assert G[u][v]["weight"] == G[v][u]["weight"]


def test_isolated_accounts_excluded(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df):
    """Accounts with zero shared resources (e.g. ACC_SOLO_01) must not be in the graph."""
    G = build_account_graph(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df)
    assert "ACC_SOLO_01" not in G.nodes


def test_compute_account_centrality_features(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df):
    """Centrality features DataFrame must have correct columns, rows, and valid ranges."""
    G = build_account_graph(synthetic_device_df, synthetic_payment_df, synthetic_address_df, synthetic_ip_df)
    features_df = compute_account_centrality_features(G, seed=42)

    expected_cols = ["account_id"] + CENTRALITY_COLUMNS
    assert list(features_df.columns) == expected_cols
    assert len(features_df) == G.number_of_nodes()

    # All centrality scores must be bounded in [0, 1]
    for col in CENTRALITY_COLUMNS:
        assert (features_df[col] >= 0.0).all()
        assert (features_df[col] <= 1.0).all()


def test_empty_graph_centrality():
    """Empty graph returns empty DataFrame with correct column structure without error."""
    G_empty = nx.Graph()
    features_df = compute_account_centrality_features(G_empty)
    assert list(features_df.columns) == ["account_id"] + CENTRALITY_COLUMNS
    assert len(features_df) == 0


def test_build_account_graph_empty_inputs():
    """Empty input tables yield an empty graph."""
    empty = pd.DataFrame(columns=["account_id", "resource_id"])
    G = build_account_graph(
        empty.rename(columns={"resource_id": "device_id"}),
        empty.rename(columns={"resource_id": "payment_id"}),
        empty.rename(columns={"resource_id": "address_id"}),
    )
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0
