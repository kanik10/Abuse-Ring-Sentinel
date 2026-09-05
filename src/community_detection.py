"""
Day 2 — community_detection.py

Runs Louvain community detection on top of graph_builder's account graph,
then evaluates the result against ground_truth.csv (evaluation-only file —
never used as a feature).

Output: clusters.csv (account_id, cluster_id, cluster_size) for Day 3's
feature engineering to consume, plus a printed precision/recall summary.
"""

from pathlib import Path

import community as community_louvain
import pandas as pd

from graph_builder import build_account_graph

DATA_DIR = "day1_data"


def load_resolved(resource: str) -> pd.DataFrame:
    """Load entity_resolution.py's output for one resource type. Raises a
    clear error instead of silently falling back to raw (exact-match) IDs
    if resolution hasn't been run yet."""
    path = Path(DATA_DIR) / f"resolved_account_{resource}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python3 entity_resolution.py` first -- "
            "the graph is meant to be built on resolved (not raw) resource IDs."
        )
    return pd.read_csv(path)


def main():
    account_device = load_resolved("device")
    account_payment = load_resolved("payment")
    account_address = load_resolved("address")
    account_ip = load_resolved("ip")
    ground_truth = pd.read_csv(f"{DATA_DIR}/ground_truth.csv")

    G = build_account_graph(account_device, account_payment, account_address, account_ip)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{__import__('networkx').number_connected_components(G)} connected components")

    # ---- Louvain ----
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)

    clusters_df = pd.DataFrame({
        "account_id": list(partition.keys()),
        "cluster_id": list(partition.values()),
    })
    cluster_sizes = clusters_df.groupby("cluster_id").size().rename("cluster_size")
    clusters_df = clusters_df.merge(cluster_sizes, on="cluster_id")

    n_communities = clusters_df["cluster_id"].nunique()
    print(f"Louvain communities: {n_communities}")

    # keep only non-trivial clusters (size >= 2) as "flagged for review"
    flagged = clusters_df[clusters_df.cluster_size >= 2].copy()
    print(f"Flagged clusters (size>=2): {flagged.cluster_id.nunique()}, "
          f"covering {len(flagged)} accounts")

    # ---- Evaluate against ground truth (evaluation only — not used above) ----
    merged = flagged.merge(ground_truth, on="account_id", how="left")

    total_ring_accounts = ground_truth.is_ring_member.sum()
    total_coincidental_accounts = ground_truth.coincidental_group_id.notna().sum()

    tp = merged.is_ring_member.sum()                       # ring accounts flagged
    fp_coincidental = merged.coincidental_group_id.notna().sum()  # benign accounts flagged
    fp_other = len(merged) - tp - fp_coincidental           # flagged, neither ring nor coincidental
    fn = total_ring_accounts - tp                           # ring accounts NOT flagged

    precision = tp / (tp + fp_coincidental + fp_other) if len(merged) else 0.0
    recall = tp / total_ring_accounts if total_ring_accounts else 0.0

    print("\n" + "=" * 60)
    print("DAY 2 SUMMARY — raw graph-structure detection (pre-classifier)")
    print("=" * 60)
    print(f"Ring accounts total:              {total_ring_accounts}")
    print(f"Coincidental accounts total:       {total_coincidental_accounts}")
    print(f"True positives (ring, flagged):    {tp}")
    print(f"False positives (coincidental):    {fp_coincidental}")
    print(f"False positives (other/unexpected):{fp_other}")
    print(f"False negatives (ring, missed):    {fn}")
    print(f"Precision:                         {precision:.3f}")
    print(f"Recall:                            {recall:.3f}")
    print()
    print("Interpretation: this is 'flag everyone in any non-trivial cluster' —")
    print("no scoring or ranking yet. Precision is capped by the fact that pure")
    print("graph structure can't yet tell a family sharing an address apart from")
    print("a ring sharing a device. That's exactly what Day 3's classifier is for.")

    # how many rings got captured mostly intact vs. split across multiple clusters?
    ring_rows = merged[merged.is_ring_member == True]
    ring_purity = ring_rows.groupby("ring_id")["cluster_id"].nunique()
    print(f"\nRings split across >1 Louvain cluster: {(ring_purity > 1).sum()} of "
          f"{ring_purity.shape[0]}")

    clusters_df.to_csv("clusters.csv", index=False)
    print("\nSaved clusters.csv (account_id, cluster_id, cluster_size) for Day 3.")


if __name__ == "__main__":
    main()
