"""
entity_resolution.py

Real-world device fingerprints, address strings, and payment tokens don't
always match exactly across observations — the same real thing can show up
as slightly different strings. This step reconstructs likely-same-entity
clusters from the messy observed data using string similarity, BEFORE the
graph is built. graph_builder.py itself stays completely unchanged — it
just receives resolved IDs instead of raw exact-match IDs.

raw_to_true_resource.csv is used ONLY to report resolution quality
afterward — never as an input to the resolution algorithm.
"""

import difflib
from pathlib import Path

import pandas as pd

DATA_DIR = Path("day1_data")
SIMILARITY_THRESHOLD = 0.85


def resolve(observed_ids: list[str], window: int = 8) -> dict:
    """Sorted-neighborhood entity resolution: sort all unique observed
    strings, then only compare each one to its next `window` neighbors in
    sorted order, unioning matches above the similarity threshold.

    This is the standard scalable alternative to naive O(n^2) pairwise
    matching, and to prefix-based blocking (which fails here — every ID
    shares a fixed "DEV"/"PAY"/"ADDR" prefix, so a prefix block would just
    contain almost everything). A single-character edit keeps two strings
    close in sort order in the overwhelming majority of cases, so a small
    window catches most true matches at a fraction of the cost. Heavily
    perturbed strings that sort far from their true match will fail to
    resolve — a real, named limitation of this approach, not a bug."""
    unique_ids = sorted(set(observed_ids))
    parent = {s: s for s in unique_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    n = len(unique_ids)
    for i in range(n):
        for j in range(i + 1, min(i + window, n)):
            a, b = unique_ids[i], unique_ids[j]
            if abs(len(a) - len(b)) > 1:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= SIMILARITY_THRESHOLD:
                union(a, b)

    return {s: find(s) for s in unique_ids}


def resolve_mapping_table(df: pd.DataFrame, resource_col: str) -> pd.DataFrame:
    mapping = resolve(df[resource_col].tolist())
    out = df.copy()
    out[resource_col] = out[resource_col].map(mapping)
    return out


if __name__ == "__main__":
    accounts = pd.read_csv(DATA_DIR / "accounts.csv")
    device = pd.read_csv(DATA_DIR / "account_device.csv")
    payment = pd.read_csv(DATA_DIR / "account_payment.csv")
    address = pd.read_csv(DATA_DIR / "account_address.csv")
    ip = pd.read_csv(DATA_DIR / "account_ip.csv")
    raw_to_true = pd.read_csv(DATA_DIR / "raw_to_true_resource.csv")

    resolved_device = resolve_mapping_table(device, "device_id")
    resolved_payment = resolve_mapping_table(payment, "payment_id")
    resolved_address = resolve_mapping_table(address, "address_id")
    resolved_ip = resolve_mapping_table(ip, "ip_id")

    print("=" * 60)
    print("ENTITY RESOLUTION QUALITY (evaluation only)")
    print("=" * 60)

    for name, raw_df, resolved_df, col in [
        ("device", device, resolved_device, "device_id"),
        ("payment", payment, resolved_payment, "payment_id"),
        ("address", address, resolved_address, "address_id"),
        ("ip", ip, resolved_ip, "ip_id"),
    ]:
        truth = raw_to_true[raw_to_true.resource_type == name]
        truth_map = dict(zip(truth.observed, truth.true_id))

        true_ids = raw_df[col].map(truth_map)
        resolved_ids = resolved_df[col]

        # for each TRUE resource, did all its observed variants land in one resolved cluster?
        combo = pd.DataFrame({"true_id": true_ids, "resolved_id": resolved_ids})
        purity = combo.groupby("true_id")["resolved_id"].nunique()
        correctly_unified = (purity == 1).sum()
        total_true = purity.shape[0]

        # did two DIFFERENT true resources ever get merged into the same resolved cluster?
        contamination = combo.groupby("resolved_id")["true_id"].nunique()
        false_merges = (contamination > 1).sum()

        print(f"{name:8s}: {correctly_unified}/{total_true} true resources fully reunified "
              f"({correctly_unified/total_true:.1%}), {false_merges} false-merge clusters")

    resolved_device.to_csv(DATA_DIR / "resolved_account_device.csv", index=False)
    resolved_payment.to_csv(DATA_DIR / "resolved_account_payment.csv", index=False)
    resolved_address.to_csv(DATA_DIR / "resolved_account_address.csv", index=False)
    resolved_ip.to_csv(DATA_DIR / "resolved_account_ip.csv", index=False)
    print(f"\nSaved resolved_account_{{device,payment,address,ip}}.csv to {DATA_DIR}/ for graph construction.")
