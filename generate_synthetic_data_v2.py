"""
generate_synthetic_data_v2.py

Realism upgrades over v1:
  1. Resources are observed as slightly messy strings (typo-like variation),
     not clean exact-match IDs -> requires real entity resolution before
     graph construction, instead of assuming clean joins.
  2. A fraction of each ring's members are "sleeper" accounts, created
     months before the ring's active burst -> defeats naive burstiness.
  3. A handful of deliberate cross-links bridge two rings, or a ring and an
     innocent bystander, together -> gives Louvain real disentangling work
     instead of matching trivial connected components.
  4. Ring order behavior is a MIX of "blends in" and "obviously cheap"
     accounts, not a uniformly low mean -> closes the avg_order_amount
     shortcut from v1.

`raw_to_true_resource.csv` keeps the perturbation->true-ID mapping for
evaluating entity resolution quality ONLY — it must never be used by the
resolution algorithm itself, same rule as ground_truth.csv for detection.
"""

import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
from faker import Faker

# SYNTH_SEED env var lets multi_seed_eval.py drive independent runs without
# touching any other line of this generator. Unset -> identical to before (42).
SEED = int(os.environ.get("SYNTH_SEED", 42))
N_LEGIT = 6000
N_RINGS = 20
RING_SIZE_MIN, RING_SIZE_MAX = 5, 40
N_COINCIDENTAL_GROUPS = 50
COINC_GROUP_MIN, COINC_GROUP_MAX = 2, 4
START_DATE = pd.Timestamp("2024-09-01")
END_DATE = pd.Timestamp("2026-08-30")
SLEEPER_FRACTION = 0.25          # fraction of each ring planted early
SLEEPER_MIN_DAYS, SLEEPER_MAX_DAYS = 60, 400  # how early, relative to burst start
N_BRIDGES = 7                    # cross-linking events
PERTURB_PROB = 0.40              # chance a resource usage is observed "messy"
RING_BLEND_IN_PROB = 0.45        # fraction of ring accounts that spend normally
OUTPUT_DIR = Path(os.environ.get("SYNTH_OUTPUT_DIR", "day1_data"))

random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]
PRODUCTS = ["Electronics", "Fashion", "Home", "Grocery", "Beauty"]
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"

def new_id(kind: str) -> str:
    """Random tokens, not sequential counters — sequential zero-padded IDs
    make two UNRELATED resources look similar by construction (e.g.
    'DEV003456' vs 'DEV003457' score 0.89 on SequenceMatcher with zero
    perturbation applied), which silently breaks string-similarity-based
    resolution downstream. Random tokens keep unrelated resources genuinely
    dissimilar, so only real perturbations of the SAME token score high."""
    prefix = {"account": "ACC", "device": "DEV", "payment": "PAY",
              "address": "ADDR", "ip": "IP", "order": "ORD"}[kind]
    token = "".join(random.choices("0123456789abcdef", k=10))
    return f"{prefix}{token}"


def random_date(start, end):
    delta_days = max((end - start).days, 0)
    return start + pd.Timedelta(days=random.randint(0, delta_days))


def perturb(s: str) -> str:
    """Simulate a messy real-world observation of an otherwise-stable
    identifier (a device fingerprint reading slightly differently between
    sessions, an address typed with a small variation)."""
    if random.random() > PERTURB_PROB:
        return s
    s = list(s)
    op = random.choice(["swap", "delete", "insert", "case"])
    if not s:
        return "".join(s)
    idx = random.randrange(len(s))
    if op == "swap" and len(s) > 1:
        j = min(idx + 1, len(s) - 1)
        s[idx], s[j] = s[j], s[idx]
    elif op == "delete" and len(s) > 4:
        del s[idx]
    elif op == "insert":
        s.insert(idx, random.choice(CHARS))
    elif op == "case":
        s[idx] = s[idx].swapcase()
    return "".join(s)


# ---------------------------------------------------------------- #
# storage
# ---------------------------------------------------------------- #
accounts = []
account_device = []       # observed (perturbed) strings — what the pipeline sees
account_payment = []
account_address = []
account_ip = []
raw_to_true = []           # evaluation-only mapping: observed string -> true resource id
ground_truth = []


def add_account_row(aid, creation_date):
    accounts.append({
        "account_id": aid, "creation_date": creation_date.date(),
        "name": fake.name(), "email_domain": random.choice(EMAIL_DOMAINS),
    })


def record_usage(mapping_list, resource_type, aid, true_id):
    observed = perturb(true_id)
    mapping_list.append({"account_id": aid, "resource_id": observed})
    raw_to_true.append({"resource_type": resource_type, "observed": observed, "true_id": true_id})


# ---------------------------------------------------------------- #
# 1. legit accounts
# ---------------------------------------------------------------- #
legit_ids = []
for _ in range(N_LEGIT):
    aid = new_id("account")
    legit_ids.append(aid)
    creation = random_date(START_DATE, END_DATE)
    add_account_row(aid, creation)

    record_usage(account_device, "device", aid, new_id("device"))
    record_usage(account_ip, "ip", aid, new_id("ip"))
    for _ in range(random.randint(1, 2)):
        record_usage(account_payment, "payment", aid, new_id("payment"))
    for _ in range(random.randint(1, 2)):
        record_usage(account_address, "address", aid, new_id("address"))

    ground_truth.append({"account_id": aid, "ring_id": None,
                          "is_ring_member": False, "coincidental_group_id": None})

gt_index = {row["account_id"]: row for row in ground_truth}

# ---------------------------------------------------------------- #
# 2. coincidental (benign) overlap groups
# ---------------------------------------------------------------- #
pool = list(legit_ids)
random.shuffle(pool)
i = 0
coincidental_resources = {}  # group_id -> (type, true_id, members)
for g in range(N_COINCIDENTAL_GROUPS):
    size = random.randint(COINC_GROUP_MIN, COINC_GROUP_MAX)
    if i + size > len(pool):
        break
    members = pool[i:i + size]
    i += size
    group_id = f"COINC{g:04d}"
    share_type = random.choice(["address", "device", "payment", "ip"])
    true_shared = new_id(share_type)
    coincidental_resources[group_id] = (share_type, true_shared, members)

    target = {"address": account_address, "device": account_device,
              "payment": account_payment, "ip": account_ip}[share_type]
    for m in members:
        record_usage(target, share_type, m, true_shared)
    for m in members:
        gt_index[m]["coincidental_group_id"] = group_id

# ---------------------------------------------------------------- #
# 3. rings — with sleepers and blend-in spending behavior
# ---------------------------------------------------------------- #
ring_member_ids = set()
ring_behavior = {}   # account_id -> "blend_in" | "suspicious"
ring_resource_pools = {}  # ring_id -> dict of true resource pools (for bridging)

for r in range(N_RINGS):
    ring_id = f"RING{r:03d}"
    ring_size = random.randint(RING_SIZE_MIN, RING_SIZE_MAX)

    ring_devices = [new_id("device") for _ in range(random.randint(2, 4))]
    ring_ips = [new_id("ip") for _ in range(random.randint(2, 4))]
    ring_payments = [new_id("payment") for _ in range(random.randint(2, 3))]
    ring_addresses = [new_id("address") for _ in range(random.randint(2, 3))]
    ring_resource_pools[ring_id] = {"device": ring_devices, "payment": ring_payments,
                                    "address": ring_addresses, "ip": ring_ips}

    ring_start = random_date(START_DATE, END_DATE - pd.Timedelta(days=30))
    ring_window_days = random.randint(5, 25)
    n_sleepers = int(ring_size * SLEEPER_FRACTION)

    for member_idx in range(ring_size):
        aid = new_id("account")
        ring_member_ids.add(aid)

        if member_idx < n_sleepers:
            offset = random.randint(SLEEPER_MIN_DAYS, SLEEPER_MAX_DAYS)
            creation = ring_start - pd.Timedelta(days=offset)
            if creation < START_DATE:
                creation = START_DATE + pd.Timedelta(days=random.randint(0, 30))
        else:
            creation = ring_start + pd.Timedelta(days=random.randint(0, ring_window_days))

        add_account_row(aid, creation)

        record_usage(account_device, "device", aid, random.choice(ring_devices))
        record_usage(account_ip, "ip", aid, random.choice(ring_ips))
        if random.random() < 0.30:
            record_usage(account_payment, "payment", aid, new_id("payment"))
        else:
            record_usage(account_payment, "payment", aid, random.choice(ring_payments))
        record_usage(account_address, "address", aid, random.choice(ring_addresses))

        ring_behavior[aid] = "blend_in" if random.random() < RING_BLEND_IN_PROB else "suspicious"

        ground_truth.append({"account_id": aid, "ring_id": ring_id,
                              "is_ring_member": True, "coincidental_group_id": None})

gt_index = {row["account_id"]: row for row in ground_truth}

# ---------------------------------------------------------------- #
# 4. bridges — deliberate cross-links for Louvain to actually resolve
# ---------------------------------------------------------------- #
ring_ids_list = list(ring_resource_pools.keys())
coincidental_ids_list = list(coincidental_resources.keys())
bridge_log = []

for b in range(N_BRIDGES):
    bridge_type = random.choice(["ring_to_ring", "ring_to_coincidental"])
    if bridge_type == "ring_to_ring" and len(ring_ids_list) >= 2:
        ring_a, ring_b = random.sample(ring_ids_list, 2)
        rtype = random.choice(["device", "payment", "address", "ip"])
        shared_true_id = random.choice(ring_resource_pools[ring_a][rtype])
        member_b = random.choice([aid for aid, row in gt_index.items() if row["ring_id"] == ring_b])
        target = {"address": account_address, "device": account_device,
                  "payment": account_payment, "ip": account_ip}[rtype]
        record_usage(target, rtype, member_b, shared_true_id)
        bridge_log.append({"type": "ring_to_ring", "a": ring_a, "b": ring_b,
                            "resource_type": rtype, "bridged_account": member_b})
    elif coincidental_ids_list and ring_ids_list:
        ring_a = random.choice(ring_ids_list)
        coinc_g = random.choice(coincidental_ids_list)
        rtype = random.choice(["device", "payment", "address", "ip"])
        shared_true_id = random.choice(ring_resource_pools[ring_a][rtype])
        _, _, coinc_members = coincidental_resources[coinc_g]
        bystander = random.choice(coinc_members)
        target = {"address": account_address, "device": account_device,
                  "payment": account_payment, "ip": account_ip}[rtype]
        record_usage(target, rtype, bystander, shared_true_id)
        bridge_log.append({"type": "ring_to_coincidental", "ring": ring_a, "group": coinc_g,
                            "resource_type": rtype, "bridged_account": bystander})

# ---------------------------------------------------------------- #
# 5. orders — mixed ring behavior (blend-in vs suspicious)
# ---------------------------------------------------------------- #
orders = []
accounts_df_tmp = pd.DataFrame(accounts)
creation_map = dict(zip(accounts_df_tmp.account_id, pd.to_datetime(accounts_df_tmp.creation_date)))

for aid, creation in creation_map.items():
    if aid in ring_member_ids:
        n_orders = random.randint(1, 3)
        amount_mean = 700 if ring_behavior[aid] == "blend_in" else 350
    else:
        n_orders = np.random.poisson(3)
        amount_mean = 800

    for _ in range(n_orders):
        order_date = creation + pd.Timedelta(days=random.randint(0, 60))
        if order_date > END_DATE:
            order_date = END_DATE
        amount = max(50.0, np.random.lognormal(mean=np.log(amount_mean), sigma=0.5))
        orders.append({
            "order_id": new_id("order"), "account_id": aid,
            "amount": round(float(amount), 2), "timestamp": order_date,
            "product": random.choice(PRODUCTS),
        })

# ---------------------------------------------------------------- #
# 6. write everything
# ---------------------------------------------------------------- #
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def usage_df(mapping_list, resource_col_name):
    df = pd.DataFrame(mapping_list)
    return df.rename(columns={"resource_id": resource_col_name})

pd.DataFrame(accounts).to_csv(OUTPUT_DIR / "accounts.csv", index=False)
usage_df(account_device, "device_id").to_csv(OUTPUT_DIR / "account_device.csv", index=False)
usage_df(account_payment, "payment_id").to_csv(OUTPUT_DIR / "account_payment.csv", index=False)
usage_df(account_address, "address_id").to_csv(OUTPUT_DIR / "account_address.csv", index=False)
usage_df(account_ip, "ip_id").to_csv(OUTPUT_DIR / "account_ip.csv", index=False)
pd.DataFrame(orders).to_csv(OUTPUT_DIR / "orders.csv", index=False)
pd.DataFrame(ground_truth).to_csv(OUTPUT_DIR / "ground_truth.csv", index=False)
pd.DataFrame(raw_to_true).to_csv(OUTPUT_DIR / "raw_to_true_resource.csv", index=False)
pd.DataFrame(bridge_log).to_csv(OUTPUT_DIR / "bridge_log.csv", index=False)

print("=" * 60)
print("V2 DATA GENERATION SUMMARY")
print("=" * 60)
print(f"Accounts: {len(accounts)}  (legit {N_LEGIT}, ring {len(ring_member_ids)})")
print(f"Rings: {N_RINGS}, sleeper fraction {SLEEPER_FRACTION}")
print(f"Coincidental groups: {N_COINCIDENTAL_GROUPS}")
print(f"Bridges (cross-links): {len(bridge_log)}")
print(f"Blend-in ring accounts: {sum(1 for v in ring_behavior.values() if v == 'blend_in')} "
      f"of {len(ring_behavior)}")
rtt = pd.DataFrame(raw_to_true)
print(f"Total resource usages: {len(rtt)}, "
      f"perturbed (messy) observations: {(rtt.observed != rtt.true_id).sum()} "
      f"({(rtt.observed != rtt.true_id).mean():.1%})")
print(f"IP usages: {len(account_ip)}")
print(f"Wrote generated CSVs to: {OUTPUT_DIR}")
