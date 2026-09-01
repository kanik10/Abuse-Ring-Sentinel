"""
Day 2 — graph_builder.py

Builds a weighted account-account graph from three account-resource
mapping tables (device, payment, address). Two accounts get an edge if
they share at least one resource; the edge weight accumulates
1 / degree(resource) across every shared resource, so a resource used by
many accounts (e.g. a shared apartment-complex address) contributes less
per pair than a resource used by only two or three accounts.

Only accounts that share at least one resource with someone else end up
in the graph — accounts with zero overlap are, by construction, not part
of any cluster and are excluded before community detection ever runs.
"""

import itertools
from collections import defaultdict

import networkx as nx
import pandas as pd

CENTRALITY_COLUMNS = [
    "degree_centrality",
    "pagerank",
    "betweenness_centrality",
]


def _add_edges_from_mapping(edge_weights: dict, mapping_df: pd.DataFrame, resource_col: str) -> None:
    """For each resource, connect every pair of accounts sharing it,
    weighted by 1 / (number of accounts sharing that resource)."""
    grouped = mapping_df.groupby(resource_col)["account_id"].apply(list)
    for accounts in grouped:
        degree = len(accounts)
        if degree < 2:
            continue  # unique to one account -> no signal
        w = 1.0 / degree
        for a, b in itertools.combinations(sorted(set(accounts)), 2):
            key = (a, b) if a < b else (b, a)
            edge_weights[key] += w


def build_account_graph(account_device: pd.DataFrame,
                         account_payment: pd.DataFrame,
                         account_address: pd.DataFrame) -> nx.Graph:
    """Returns a weighted, undirected NetworkX graph over accounts that
    share at least one device/payment instrument/address with another
    account. Isolated accounts (no sharing at all) are not added."""
    edge_weights = defaultdict(float)

    _add_edges_from_mapping(edge_weights, account_device, "device_id")
    _add_edges_from_mapping(edge_weights, account_payment, "payment_id")
    _add_edges_from_mapping(edge_weights, account_address, "address_id")

    G = nx.Graph()
    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=w)

    return G


def compute_account_centrality_features(G: nx.Graph,
                                        betweenness_k: int = 100,
                                        seed: int = 42) -> pd.DataFrame:
    """Returns per-account centrality features on the account-account graph.

    build_account_graph already returns the account-account projection:
    accounts are nodes, and two accounts are connected when they share at
    least one device/payment/address. Centrality is intentionally computed
    on that projection rather than on account-resource tables.
    """
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return pd.DataFrame(columns=["account_id"] + CENTRALITY_COLUMNS)

    degree = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")

    # The current generated graph has 531 account nodes. Sampling up to 100
    # keeps betweenness tractable as the graph grows while still covering a
    # meaningful share of this dataset; seed=42 keeps the approximation stable.
    k = min(betweenness_k, n_nodes)
    betweenness = nx.betweenness_centrality(G, k=k, seed=seed, weight=None)

    rows = []
    for account_id in G.nodes:
        rows.append({
            "account_id": account_id,
            "degree_centrality": degree.get(account_id, 0.0),
            "pagerank": pagerank.get(account_id, 0.0),
            "betweenness_centrality": betweenness.get(account_id, 0.0),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    account_device = pd.read_csv("day1_data/account_device.csv")
    account_payment = pd.read_csv("day1_data/account_payment.csv")
    account_address = pd.read_csv("day1_data/account_address.csv")

    G = build_account_graph(account_device, account_payment, account_address)
    print(f"Graph nodes (accounts with >=1 shared resource): {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    print(f"Largest 10 component sizes: {sizes[:10]}")
    centrality = compute_account_centrality_features(G)
    print(f"Computed account centrality features: {centrality.shape[1] - 1} per account")
