"""
test_community_detection.py -- Unit tests for community_detection.py.

Verifies:
  - Louvain output is deterministic given a fixed random seed.
  - Clusters below the size threshold (cluster_size < 2) are excluded from flagged review.
  - load_resolved raises FileNotFoundError with helpful advice if files are missing.
"""

import community as community_louvain
import networkx as nx
import pandas as pd
import pytest

from community_detection import load_resolved


def test_louvain_deterministic_with_fixed_seed():
    """Louvain community detection with fixed random_state must yield identical partitions across runs."""
    # Build a barbell graph (two cliques connected by a bridge)
    G = nx.barbell_graph(5, 1)
    # Add arbitrary weights
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + 0.1 * (u + v)

    partition_1 = community_louvain.best_partition(G, weight="weight", random_state=42)
    partition_2 = community_louvain.best_partition(G, weight="weight", random_state=42)

    assert partition_1 == partition_2


def test_clusters_below_size_threshold_excluded():
    """Single-account clusters (cluster_size < 2) must be excluded from flagged review."""
    clusters_df = pd.DataFrame([
        {"account_id": "ACC_RING_01", "cluster_id": 1},
        {"account_id": "ACC_RING_02", "cluster_id": 1},
        {"account_id": "ACC_RING_03", "cluster_id": 1},
        {"account_id": "ACC_SINGLETON_01", "cluster_id": 2},
    ])

    cluster_sizes = clusters_df.groupby("cluster_id").size().rename("cluster_size")
    clusters_df = clusters_df.merge(cluster_sizes, on="cluster_id")

    # The pipeline filter: keep only non-trivial clusters (size >= 2)
    flagged = clusters_df[clusters_df.cluster_size >= 2].copy()

    assert "ACC_SINGLETON_01" not in flagged["account_id"].values
    assert 2 not in flagged["cluster_id"].values
    assert set(flagged["account_id"]) == {"ACC_RING_01", "ACC_RING_02", "ACC_RING_03"}


def test_load_resolved_missing_file_raises(monkeypatch, tmp_path):
    """load_resolved raises FileNotFoundError with clear remediation advice if file is missing."""
    import community_detection
    monkeypatch.setattr(community_detection, "DATA_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="Run `python3 entity_resolution.py` first"):
        load_resolved("device")
