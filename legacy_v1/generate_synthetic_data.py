"""
Day 1 — Abuse-Ring Sentinel: synthetic data generator.

Produces a population of accounts where:
  - most accounts are independent, ordinary customers ("legit")
  - a small number of accounts are organized into abuse rings that share
    a rotating pool of devices/payment instruments/addresses, staggered
    over time (camouflaged, not trivially identical)
  - a separate small number of legit accounts coincidentally share a
    resource (family sharing an address, roommates sharing a device) —
    these are the hard negatives that make the false-positive analysis honest

Ground truth (ring_id / is_ring_member / coincidental_group_id) is written
to its own file and must NEVER be joined into the feature-engineering step
on days 2-3 — it exists only for evaluation on day 4.
"""

import random
import pandas as pd
import numpy as np
from faker import Faker

# ---------------------------------------------------------------- #
# Config — tune these, but keep a record of whatever you change here
# ---------------------------------------------------------------- #
SEED = 42
N_LEGIT = 6000
N_RINGS = 20
RING_SIZE_MIN, RING_SIZE_MAX = 5, 40
N_COINCIDENTAL_GROUPS = 50
COINC_GROUP_MIN, COINC_GROUP_MAX = 2, 4
START_DATE = pd.Timestamp("2024-09-01")
END_DATE = pd.Timestamp("2026-08-29")
RING_PERSONAL_PAYMENT_PROB = 0.30  # fraction of ring members who use a unique
                                    # payment instrument instead of a shared one
                                    # (camouflage noise)

random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]
PRODUCTS = ["Electronics", "Fashion", "Home", "Grocery", "Beauty"]

# ---------------------------------------------------------------- #
# ID generators
# ---------------------------------------------------------------- #
_counters = {"account": 0, "device": 0, "payment": 0, "address": 0, "order": 0}


def new_id(kind: str) -> str:
    _counters[kind] += 1
    prefix = {"account": "ACC", "device": "DEV", "payment": "PAY",
              "address": "ADDR", "order": "ORD"}[kind]
    width = 7 if kind == "order" else 6
    return f"{prefix}{_counters[kind]:0{width}d}"


def random_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    delta_days = max((end - start).days, 0)
    return start + pd.Timedelta(days=random.randint(0, delta_days))


# ---------------------------------------------------------------- #
# Storage
# ---------------------------------------------------------------- #
accounts = []
account_device = []
account_payment = []
account_address = []
ground_truth = []


def add_account_row(aid, creation_date):
    accounts.append({
        "account_id": aid,
        "creation_date": creation_date.date(),
        "name": fake.name(),
        "email_domain": random.choice(EMAIL_DOMAINS),
    })


# ---------------------------------------------------------------- #
# 1. Legit accounts — independent, ordinary behavior
# ---------------------------------------------------------------- #
legit_ids = []
for _ in range(N_LEGIT):
    aid = new_id("account")
    legit_ids.append(aid)
    creation = random_date(START_DATE, END_DATE)
    add_account_row(aid, creation)

    account_device.append({"account_id": aid, "device_id": new_id("device")})
    for _ in range(random.randint(1, 2)):
        account_payment.append({"account_id": aid, "payment_id": new_id("payment")})
    for _ in range(random.randint(1, 2)):
        account_address.append({"account_id": aid, "address_id": new_id("address")})

    ground_truth.append({
        "account_id": aid, "ring_id": None,
        "is_ring_member": False, "coincidental_group_id": None,
    })

gt_index = {row["account_id"]: row for row in ground_truth}

# ---------------------------------------------------------------- #
# 2. Coincidental overlap groups — benign sharing, NOT abuse
#    (family address, shared device, shared family card)
# ---------------------------------------------------------------- #
pool = list(legit_ids)
random.shuffle(pool)
i = 0
for g in range(N_COINCIDENTAL_GROUPS):
    size = random.randint(COINC_GROUP_MIN, COINC_GROUP_MAX)
    if i + size > len(pool):
        break
    members = pool[i:i + size]
    i += size
    group_id = f"COINC{g:04d}"
    share_type = random.choice(["address", "device", "payment"])

    if share_type == "address":
        shared = new_id("address")
        for m in members:
            account_address.append({"account_id": m, "address_id": shared})
    elif share_type == "device":
        shared = new_id("device")
        for m in members:
            account_device.append({"account_id": m, "device_id": shared})
    else:
        shared = new_id("payment")
        for m in members:
            account_payment.append({"account_id": m, "payment_id": shared})

    for m in members:
        gt_index[m]["coincidental_group_id"] = group_id

# ---------------------------------------------------------------- #
# 3. Ring accounts — camouflaged abuse rings
# ---------------------------------------------------------------- #
ring_member_ids = set()
for r in range(N_RINGS):
    ring_id = f"RING{r:03d}"
    ring_size = random.randint(RING_SIZE_MIN, RING_SIZE_MAX)

    # small rotating resource pools — NOT one identical set per member
    ring_devices = [new_id("device") for _ in range(random.randint(2, 4))]
    ring_payments = [new_id("payment") for _ in range(random.randint(2, 3))]
    ring_addresses = [new_id("address") for _ in range(random.randint(2, 3))]

    # staggered creation window (days, not one instant burst)
    ring_start = random_date(START_DATE, END_DATE - pd.Timedelta(days=30))
    ring_window_days = random.randint(5, 25)

    for _ in range(ring_size):
        aid = new_id("account")
        ring_member_ids.add(aid)
        creation = ring_start + pd.Timedelta(days=random.randint(0, ring_window_days))
        add_account_row(aid, creation)

        account_device.append({"account_id": aid, "device_id": random.choice(ring_devices)})

        if random.random() < RING_PERSONAL_PAYMENT_PROB:
            account_payment.append({"account_id": aid, "payment_id": new_id("payment")})
        else:
            account_payment.append({"account_id": aid, "payment_id": random.choice(ring_payments)})

        account_address.append({"account_id": aid, "address_id": random.choice(ring_addresses)})

        ground_truth.append({
            "account_id": aid, "ring_id": ring_id,
            "is_ring_member": True, "coincidental_group_id": None,
        })

# rebuild lookup (ring accounts were appended after legit ones)
gt_index = {row["account_id"]: row for row in ground_truth}

# ---------------------------------------------------------------- #
# 4. Orders — ring accounts skew toward smaller, threshold-chasing orders
# ---------------------------------------------------------------- #
orders = []
accounts_df_tmp = pd.DataFrame(accounts)
creation_map = dict(zip(accounts_df_tmp.account_id, pd.to_datetime(accounts_df_tmp.creation_date)))

for aid, creation in creation_map.items():
    if aid in ring_member_ids:
        n_orders = random.randint(1, 3)
        amount_mean = 350
    else:
        n_orders = np.random.poisson(3)
        amount_mean = 800

    for _ in range(n_orders):
        order_date = creation + pd.Timedelta(days=random.randint(0, 60))
        if order_date > END_DATE:
            order_date = END_DATE
        amount = max(50.0, np.random.lognormal(mean=np.log(amount_mean), sigma=0.5))
        orders.append({
            "order_id": new_id("order"),
            "account_id": aid,
            "amount": round(float(amount), 2),
            "timestamp": order_date,
            "product": random.choice(PRODUCTS),
        })

# ---------------------------------------------------------------- #
# 5. Write everything out
# ---------------------------------------------------------------- #
pd.DataFrame(accounts).to_csv("accounts.csv", index=False)
pd.DataFrame(account_device).to_csv("account_device.csv", index=False)
pd.DataFrame(account_payment).to_csv("account_payment.csv", index=False)
pd.DataFrame(account_address).to_csv("account_address.csv", index=False)
pd.DataFrame(orders).to_csv("orders.csv", index=False)
pd.DataFrame(ground_truth).to_csv("ground_truth.csv", index=False)

# ---------------------------------------------------------------- #
# 6. Sanity-check summary — read this before moving to Day 2
# ---------------------------------------------------------------- #
gt_df = pd.DataFrame(ground_truth)
dev_df = pd.DataFrame(account_device)
addr_df = pd.DataFrame(account_address)
pay_df = pd.DataFrame(account_payment)

print("=" * 60)
print("DAY 1 SUMMARY")
print("=" * 60)
print(f"Total accounts:            {len(accounts)}")
print(f"  Legit accounts:          {N_LEGIT}")
print(f"  Ring accounts:           {len(ring_member_ids)}  across {N_RINGS} rings")
print(f"  Coincidental-group accts:{gt_df['coincidental_group_id'].notna().sum()}"
      f"  across {gt_df['coincidental_group_id'].nunique()} groups")
print(f"Total orders:              {len(orders)}")
print()
print("Ring size distribution:")
print(gt_df[gt_df.is_ring_member].groupby("ring_id").size().describe()[["min", "mean", "max"]])
print()
print("Device sharing — degree distribution (accounts per device):")
print(dev_df.groupby("device_id").size().describe()[["min", "mean", "max"]])
print()
print("Address sharing — degree distribution (accounts per address):")
print(addr_df.groupby("address_id").size().describe()[["min", "mean", "max"]])
print()
print("Payment sharing — degree distribution (accounts per payment instrument):")
print(pay_df.groupby("payment_id").size().describe()[["min", "mean", "max"]])
print()
print("Files written: accounts.csv, account_device.csv, account_payment.csv,")
print("               account_address.csv, orders.csv, ground_truth.csv")
