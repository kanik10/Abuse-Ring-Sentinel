"""
temporal_reconstruction.py -- Point-in-time / as-of-T reconstruction.

PHASE 1 (this file, so far): the entity-based ring<->cluster lookup that
Phase 2's snapshot engine will depend on. Phase 2 will add the actual
as-of-T graph/feature/score reconstruction on top of this.

--- Why this function has to exist at all ---
Louvain community detection provides NO guarantee that cluster_id N at
snapshot T1 refers to the same accounts as cluster_id N at snapshot T2 --
cluster numbering is an artifact of that run's internal traversal order,
not a stable identity. So "track cluster 4's risk score over time" is
wrong on its face: cluster 4 might be a completely different set of
accounts at each snapshot.

The correct approach is entity-based, not ID-based: at every snapshot, ask
"which cluster currently contains ring X's members" fresh, using
ground-truth ring membership (eval-only, exactly like everywhere else in
this codebase -- never fed into features or clustering itself). This file
answers that question and nothing else; Phase 2 will call it once per
(ring, snapshot) pair and follow up with the classifier's score for
whichever cluster comes back.

find_dominant_cluster() reuses the exact same >50%-majority convention
classifier.py's label_clusters() already uses to decide whether a cluster
IS a ring cluster -- there is one definition of "ring cluster" in this
codebase, not two.
"""

from __future__ import annotations
import pandas as pd


def find_dominant_cluster(clusters: pd.DataFrame, target_member_ids: set,
                           min_capture: float = 0.5) -> dict | None:
    """
    Finds whichever cluster captures the largest share of target_member_ids.

    Parameters
    ----------
    clusters : DataFrame with columns cluster_id, account_id (one row per
        account). At snapshot time this is the point-in-time clustering,
        not necessarily the final one -- restricted to whatever accounts
        "exist" as of T (Phase 2's job to construct; this function doesn't
        care where clusters came from).
    target_member_ids : the account_ids we're trying to locate -- normally
        a ring's ground-truth members, ALREADY restricted by the caller to
        only the members that exist as of the snapshot being evaluated
        (this function does not know about time at all; that's Phase 2's
        job, kept out of this function deliberately so it's testable in
        isolation).
    min_capture : minimum fraction of target_member_ids that must land in
        one cluster for that cluster to count as "the" cluster. Default
        0.5 matches classifier.py's label_clusters() majority convention.

    Returns
    -------
    None if target_member_ids is empty, none of them appear in `clusters`
    at all (nothing to find -- the correct answer very early in a ring's
    life, before any member has been clustered with anyone), or no single
    cluster reaches min_capture (the ring's early members are scattered
    across multiple clusters with no dominant one yet -- also a correct,
    informative answer, not a bug to work around).

    Otherwise a dict:
      cluster_id        -- the dominant cluster's id (only meaningful
                            within this one snapshot -- see module docstring)
      cluster_size       -- total accounts in that cluster at this snapshot
      members_captured   -- how many target_member_ids are in it
      target_member_count -- len(target_member_ids), for context
      capture_fraction   -- members_captured / target_member_count
      cluster_purity     -- members_captured / cluster_size (how much of
                            the cluster IS this ring vs. diluted by others)
    """
    if not target_member_ids:
        return None

    sub = clusters[clusters["account_id"].isin(target_member_ids)]
    if sub.empty:
        return None

    counts = sub.groupby("cluster_id")["account_id"].nunique()
    best_cluster_id = counts.idxmax()
    members_captured = int(counts.loc[best_cluster_id])
    capture_fraction = members_captured / len(target_member_ids)
    if capture_fraction < min_capture:
        return None

    cluster_size = int((clusters["cluster_id"] == best_cluster_id).sum())
    return {
        "cluster_id": best_cluster_id,
        "cluster_size": cluster_size,
        "members_captured": members_captured,
        "target_member_count": len(target_member_ids),
        "capture_fraction": round(capture_fraction, 4),
        "cluster_purity": round(members_captured / cluster_size, 4) if cluster_size else 0.0,
    }


if __name__ == "__main__":
    # --- Part 1: synthetic unit tests (no file I/O) -- exercise the edge
    # cases real data won't reliably hit on its own: empty input, total
    # absence, a clean majority, and a genuinely split/ambiguous ring. ---
    print("=" * 60)
    print("UNIT TESTS (synthetic, no file I/O)")
    print("=" * 60)

    toy = pd.DataFrame({
        "cluster_id": [0, 0, 0, 0, 1, 1, 1, 2, 2],
        "account_id": ["A1", "A2", "A3", "X1", "A4", "A5", "X2", "X3", "X4"],
    })

    # Case 1: empty target set -> None
    assert find_dominant_cluster(toy, set()) is None
    print("[OK] empty target_member_ids -> None")

    # Case 2: target members entirely absent from clusters -> None
    assert find_dominant_cluster(toy, {"NOTPRESENT1", "NOTPRESENT2"}) is None
    print("[OK] target members absent from clusters -> None")

    # Case 3: clean majority -- 3 of 4 ring members in cluster 0
    result = find_dominant_cluster(toy, {"A1", "A2", "A3", "A4"})
    assert result is not None
    assert result["cluster_id"] == 0
    assert result["members_captured"] == 3
    assert result["capture_fraction"] == 0.75
    assert result["cluster_purity"] == 0.75  # 3 of cluster 0's 4 members
    print(f"[OK] clean majority case -> {result}")

    # Case 4: split ring, no cluster reaches 50% -> None
    # 2 members in cluster 0, 2 in cluster 1, out of 4 total -> 50/50 split,
    # both exactly AT min_capture=0.5 by fraction but idxmax picks the
    # first-encountered on a tie; the real test here is that a genuinely
    # scattered ring (below 50% each) returns None.
    split_result = find_dominant_cluster(toy, {"A1", "A4", "X3", "X4"}, min_capture=0.5)
    # A1->cluster0 (1), A4->cluster1 (1), X3,X4->cluster2 (2) => cluster2 has 2/4 = 0.5, meets threshold
    assert split_result is not None and split_result["cluster_id"] == 2
    print(f"[OK] plurality-of-4-way-split case -> {split_result}")

    # Case 5: nobody reaches min_capture
    scattered_result = find_dominant_cluster(toy, {"A1", "A4", "X3"}, min_capture=0.6)
    # each of cluster0/1/2 gets exactly 1 of 3 = 0.333, below 0.6
    assert scattered_result is None
    print("[OK] no cluster reaches min_capture -> None")

    print("\nAll unit tests passed.\n")

    # --- Part 2: sanity check against real, already-committed data. ---
    # Full-visibility check (T = "end of time", i.e. no temporal filtering
    # at all yet -- that's Phase 2) -- every one of these 20 resource-
    # sharing rings and 8 referral rings should resolve cleanly to ONE
    # dominant cluster with high capture_fraction, because this is the
    # exact same data the classifier already gets precision=1.0/recall=1.0
    # on. If this check ever fails, that's a real bug in this function,
    # not a property of the data. ---
    print("=" * 60)
    print("SANITY CHECK against real clusters.csv / ground_truth.csv")
    print("(full visibility -- no temporal filtering, that's Phase 2)")
    print("=" * 60)

    clusters = pd.read_csv("clusters.csv")
    ground_truth = pd.read_csv("ground_truth.csv")

    ring_ids = sorted(ground_truth.loc[ground_truth["is_ring_member"] == True, "ring_id"].dropna().unique())
    n_resolved = 0
    n_unresolved = 0
    for ring_id in ring_ids:
        members = set(ground_truth.loc[ground_truth["ring_id"] == ring_id, "account_id"])
        result = find_dominant_cluster(clusters, members)
        if result is None:
            n_unresolved += 1
            print(f"  {ring_id}: {len(members)} members -- NO dominant cluster found (unexpected)")
        else:
            n_resolved += 1
            flag = "" if result["capture_fraction"] >= 0.9 else "  <-- capture_fraction below 0.9, worth a look"
            print(f"  {ring_id}: {len(members)} members -> cluster {result['cluster_id']} "
                  f"(captured {result['members_captured']}/{result['target_member_count']} = "
                  f"{result['capture_fraction']:.2f}, purity {result['cluster_purity']:.2f}){flag}")

    print(f"\n{n_resolved}/{len(ring_ids)} resource-sharing rings resolved to a dominant cluster "
          f"at full visibility.")

    # Referral rings: EXPECTED to behave differently. They deliberately
    # avoid resource sharing (only share one IP), so at full visibility
    # they may or may not form a strong dominant cluster via the
    # resource-sharing graph alone -- that's the whole premise of why
    # referral_features.py exists as a separate signal. Reporting this
    # here (not asserting success) so the difference from resource rings
    # is visible, not silently glossed over.
    rgt_path = "day1_data/referral_ground_truth.csv"
    import os
    if os.path.exists(rgt_path):
        print("\n" + "=" * 60)
        print("Referral rings (expected to differ -- see note above)")
        print("=" * 60)
        ref_gt = pd.read_csv("day1_data/ground_truth.csv") if os.path.exists("day1_data/ground_truth.csv") \
            else ground_truth
        rref_ids = sorted(
            ref_gt.loc[ref_gt.get("is_referral_ring_member", False) == True, "referral_ring_id"]
            .dropna().unique()
        )
        for rref_id in rref_ids:
            members = set(ref_gt.loc[ref_gt["referral_ring_id"] == rref_id, "account_id"])
            result = find_dominant_cluster(clusters, members, min_capture=0.5)
            if result is None:
                print(f"  {rref_id}: {len(members)} members -- no dominant resource-sharing cluster "
                      f"(expected -- referral signal, not resource-sharing signal)")
            else:
                print(f"  {rref_id}: {len(members)} members -> cluster {result['cluster_id']} "
                      f"(captured {result['members_captured']}/{result['target_member_count']} = "
                      f"{result['capture_fraction']:.2f}, purity {result['cluster_purity']:.2f})")
