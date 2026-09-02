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

# ---- Referral-ring parameters -------------------------------------------- #
# Referral rings exploit a referral-bonus program. Members share one IP (a
# weak resource-sharing signal that creates a Louvain cluster) but deliberately
# avoid sharing devices/payments/addresses to evade the resource-sharing
# detector.  The referral chain is the primary fraud fingerprint.
N_REFERRAL_RINGS = 8
REFERRAL_RING_TRUNK_FANOUT_MIN = 2  # mastermind refers this many accounts directly
REFERRAL_RING_TRUNK_FANOUT_MAX = 4
REFERRAL_CHAIN_DEPTH_LAMBDA = 2.0   # Poisson lambda for per-trunk chain depth
REFERRAL_CHAIN_DEPTH_MAX = 4        # clip depth at this value
REFERRAL_ACTIVATION_DAYS_RING = 7   # ring: referred account activates within N days
REFERRAL_ACTIVATION_DAYS_ORGANIC = 30  # organic: wider spread
REFERRAL_ACTIVATION_RATE = 0.92     # 92% of ring referrals activate (not 100%)
RING_REFERRAL_CYCLE_PROB = 0.35     # probability a ring has a closing cycle edge
N_ORGANIC_REFERRERS_FRAC = 0.15     # fraction of legit accounts that refer others
ORGANIC_REFERRAL_MAX_FANOUT = 4     # max accounts a legit referrer refers
ORGANIC_REFERRAL_ACTIVATION_RATE = 0.70  # 30% dropout — realistic
REFERRAL_BONUS_MIN, REFERRAL_BONUS_MAX = 150.0, 500.0  # Rs per referral bonus

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
account_device = []       # observed (perturbed) strings -- what the pipeline sees
account_payment = []
account_address = []
account_ip = []
raw_to_true = []           # evaluation-only mapping: observed string -> true resource id
ground_truth = []
referrals = []             # referrer_id, referred_id, referral_date, bonus_amount
referral_ground_truth_rows = []  # eval only: referrer_id, referred_id, is_ring_referral
referral_ring_member_ids = set()


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
                          "is_ring_member": False, "coincidental_group_id": None,
                          "referral_ring_id": None, "is_referral_ring_member": False})

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
                              "is_ring_member": True, "coincidental_group_id": None,
                              "referral_ring_id": None, "is_referral_ring_member": False})

gt_index = {row["account_id"]: row for row in ground_truth}

# ---------------------------------------------------------------- #
# 3b. referral rings — hybrid trunk-branch topology
#
# Each ring:  mastermind (M) refers TRUNK_FANOUT accounts; each trunk
# account refers a depth-d chain (d ~ Poisson(2), clip [1,4]).  All ring
# members share ONE ring-specific IP, creating a weak Louvain cluster
# (entity_reuse_ratio ~0.25) that the resource-sharing classifier alone
# will not flag above the threshold — the referral features close that gap.
#
# With RING_REFERRAL_CYCLE_PROB a leaf account refers back to M, creating
# a directed cycle.  This is rare in organic referral trees (DAGs) but
# deliberate in bonus-farming rings.
# ---------------------------------------------------------------- #
for r in range(N_REFERRAL_RINGS):
    rref_id = f"RREF{r:03d}"
    ring_shared_ip = new_id("ip")  # one IP shared by ALL members of this referral ring
    trunk_fanout = random.randint(REFERRAL_RING_TRUNK_FANOUT_MIN, REFERRAL_RING_TRUNK_FANOUT_MAX)

    # --- mastermind account ---
    m_aid = new_id("account")
    m_creation = random_date(START_DATE, END_DATE - pd.Timedelta(days=180))
    add_account_row(m_aid, m_creation)
    record_usage(account_device, "device", m_aid, new_id("device"))  # unique device
    record_usage(account_ip, "ip", m_aid, ring_shared_ip)            # shared IP
    record_usage(account_payment, "payment", m_aid, new_id("payment"))  # unique payment
    record_usage(account_address, "address", m_aid, new_id("address"))  # unique address
    ground_truth.append({"account_id": m_aid, "ring_id": None,
                          "is_ring_member": False, "coincidental_group_id": None,
                          "referral_ring_id": rref_id, "is_referral_ring_member": True})
    gt_index[m_aid] = ground_truth[-1]
    referral_ring_member_ids.add(m_aid)

    chain_leaves = []  # (account_id, last_referral_date) for cycle-closing

    for t in range(trunk_fanout):
        # --- trunk account (M refers this one directly) ---
        t_aid = new_id("account")
        t_ref_date = m_creation + pd.Timedelta(days=random.randint(5, 30))
        t_creation = t_ref_date - pd.Timedelta(days=random.randint(0, 3))  # created just before referral
        if t_creation < START_DATE:
            t_creation = START_DATE
        add_account_row(t_aid, t_creation)
        record_usage(account_device, "device", t_aid, new_id("device"))
        record_usage(account_ip, "ip", t_aid, ring_shared_ip)
        record_usage(account_payment, "payment", t_aid, new_id("payment"))
        record_usage(account_address, "address", t_aid, new_id("address"))
        ground_truth.append({"account_id": t_aid, "ring_id": None,
                              "is_ring_member": False, "coincidental_group_id": None,
                              "referral_ring_id": rref_id, "is_referral_ring_member": True})
        gt_index[t_aid] = ground_truth[-1]
        referral_ring_member_ids.add(t_aid)

        bonus = round(random.uniform(REFERRAL_BONUS_MIN, REFERRAL_BONUS_MAX), 2)
        referrals.append({"referrer_id": m_aid, "referred_id": t_aid,
                           "referral_date": t_ref_date, "bonus_amount": bonus})
        referral_ground_truth_rows.append({"referrer_id": m_aid, "referred_id": t_aid,
                                           "is_ring_referral": True})

        # --- chain accounts hanging from this trunk ---
        chain_depth = min(max(1, int(np.random.poisson(REFERRAL_CHAIN_DEPTH_LAMBDA))),
                          REFERRAL_CHAIN_DEPTH_MAX)
        prev_aid = t_aid
        prev_ref_date = t_ref_date

        for d in range(chain_depth):
            c_aid = new_id("account")
            c_ref_date = prev_ref_date + pd.Timedelta(days=random.randint(2, 7))
            c_creation = c_ref_date - pd.Timedelta(days=random.randint(0, 3))
            if c_creation < START_DATE:
                c_creation = START_DATE
            if c_creation > END_DATE:
                break  # out of data range
            add_account_row(c_aid, c_creation)
            record_usage(account_device, "device", c_aid, new_id("device"))
            record_usage(account_ip, "ip", c_aid, ring_shared_ip)
            record_usage(account_payment, "payment", c_aid, new_id("payment"))
            record_usage(account_address, "address", c_aid, new_id("address"))
            ground_truth.append({"account_id": c_aid, "ring_id": None,
                                  "is_ring_member": False, "coincidental_group_id": None,
                                  "referral_ring_id": rref_id, "is_referral_ring_member": True})
            gt_index[c_aid] = ground_truth[-1]
            referral_ring_member_ids.add(c_aid)

            bonus = round(random.uniform(REFERRAL_BONUS_MIN, REFERRAL_BONUS_MAX), 2)
            referrals.append({"referrer_id": prev_aid, "referred_id": c_aid,
                               "referral_date": c_ref_date, "bonus_amount": bonus})
            referral_ground_truth_rows.append({"referrer_id": prev_aid, "referred_id": c_aid,
                                               "is_ring_referral": True})
            prev_aid = c_aid
            prev_ref_date = c_ref_date

        chain_leaves.append((prev_aid, prev_ref_date))

    # --- optional cycle: last chain leaf refers back to mastermind ---
    # Use the mastermind creation date as the cycle reference to guarantee
    # it lands within the data window regardless of chain length.
    if random.random() < RING_REFERRAL_CYCLE_PROB and chain_leaves:
        leaf_aid, leaf_ref_date = random.choice(chain_leaves)
        # Cycle date = day after the leaf's referral date (guaranteed in-range
        # because c_ref_date is already capped to END_DATE)
        cycle_date = leaf_ref_date + pd.Timedelta(days=random.randint(1, 3))
        cycle_date = min(cycle_date, END_DATE)
        bonus = round(random.uniform(REFERRAL_BONUS_MIN, REFERRAL_BONUS_MAX), 2)
        referrals.append({"referrer_id": leaf_aid, "referred_id": m_aid,
                           "referral_date": cycle_date, "bonus_amount": bonus})
        referral_ground_truth_rows.append({"referrer_id": leaf_aid, "referred_id": m_aid,
                                           "is_ring_referral": True})

# Re-index gt_index after all ring and referral-ring additions
gt_index = {row["account_id"]: row for row in ground_truth}

# ---------------------------------------------------------------- #
# 3c. organic referral baseline
#
# Without organic referrals in the data, ANY referral edge is trivially
# suspicious -- the model would learn "has referral edge => ring" rather
# than learning the structural features that distinguish ring chains from
# organic ones.  15% of legit accounts refer 1-4 others, with 70%
# activation and a 0-30 day spread (vs ring's 0-7 days).
# ---------------------------------------------------------------- #
n_organic_referrers = int(N_LEGIT * N_ORGANIC_REFERRERS_FRAC)
organic_referrer_pool = random.sample(legit_ids, n_organic_referrers)
organic_referred_already = set()  # prevent an account being referred twice

for referrer_id in organic_referrer_pool:
    referrer_creation = next(
        (a["creation_date"] for a in accounts if a["account_id"] == referrer_id), None
    )
    if referrer_creation is None:
        continue
    referrer_creation = pd.Timestamp(referrer_creation)
    n_referrals = random.randint(1, ORGANIC_REFERRAL_MAX_FANOUT)
    # Sample referred accounts from legit pool, excluding referrer and already-referred
    candidates = [aid for aid in legit_ids
                  if aid != referrer_id and aid not in organic_referred_already
                  and aid not in referral_ring_member_ids]
    if len(candidates) < n_referrals:
        continue
    referred_batch = random.sample(candidates, n_referrals)
    for referred_id in referred_batch:
        organic_referred_already.add(referred_id)
        ref_date = referrer_creation + pd.Timedelta(days=random.randint(10, 200))
        if ref_date > END_DATE:
            ref_date = END_DATE
        bonus = round(random.uniform(REFERRAL_BONUS_MIN, REFERRAL_BONUS_MAX), 2)
        referrals.append({"referrer_id": referrer_id, "referred_id": referred_id,
                           "referral_date": ref_date, "bonus_amount": bonus})
        referral_ground_truth_rows.append({"referrer_id": referrer_id, "referred_id": referred_id,
                                           "is_ring_referral": False})

# ---------------------------------------------------------------- #
# 4. bridges -- deliberate cross-links for Louvain to actually resolve
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
# Referral tables
# referrals.csv  -- pipeline input (NO label column)
# referral_ground_truth.csv -- evaluation only (has is_ring_referral label)
pd.DataFrame(referrals).to_csv(OUTPUT_DIR / "referrals.csv", index=False)
pd.DataFrame(referral_ground_truth_rows).to_csv(OUTPUT_DIR / "referral_ground_truth.csv", index=False)

print("=" * 60)
print("V2 DATA GENERATION SUMMARY")
print("=" * 60)
print(f"Accounts: {len(accounts)}  (legit {N_LEGIT}, ring {len(ring_member_ids)}, "
      f"referral-ring {len(referral_ring_member_ids)})")
print(f"Resource-sharing rings: {N_RINGS}, sleeper fraction {SLEEPER_FRACTION}")
print(f"Referral rings: {N_REFERRAL_RINGS} "
      f"(referral edges: {len([r for r in referral_ground_truth_rows if r['is_ring_referral']])} ring, "
      f"{len([r for r in referral_ground_truth_rows if not r['is_ring_referral']])} organic)")
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
