# Abuse-Ring Sentinel — Complete Data Dictionary

**Dataset Version:** V2 (Multi-Entity Bipartite Resolution + Directed Referral Abuse Rings)  
**Location:** Canonical data resides in `day1_data/`  
**Population Size:** 6,558 Accounts | 18,966 Orders | 2,240 Referrals | 28 Ground-Truth Fraud Rings (20 resource-sharing + 8 referral)

---

## 1. Canonical Relational Tables (`day1_data/`)

### 1.1 `accounts.csv` (6,558 rows)
Primary account entity registry representing legitimate customers, sleeper accounts, and syndicate fraud operators.

| Column | Type | Description |
| :--- | :--- | :--- |
| `account_id` | `string` (PK) | Unique account identifier (`ACC_xxxxxx`). |
| `creation_date` | `timestamp` | Date and time the account was registered (`YYYY-MM-DD HH:MM:SS`). |
| `name` | `string` | User display / legal profile name generated via Faker. |
| `email_domain` | `string` | Domain of the registration email (e.g. `gmail.com`, `yahoo.com`, or domain rotation pools). |

---

### 1.2 Entity-Resolved Mapping Tables
These tables map accounts to underlying hardware, network, and payment rails after entity resolution (collapsing alias accounts onto true physical entities). Each link contains a `first_seen_date` enabling point-in-time temporal reconstruction.

> **Note:** `day1_data/` also contains `account_device.csv`, `account_payment.csv`, `account_address.csv`, and `account_ip.csv` — the pre-resolution counterparts of the four tables below. In the current synthetic generator these are identical to their `resolved_*` versions (no aliasing noise was introduced for this run); the pipeline reads exclusively from the `resolved_*` tables.

#### `resolved_account_device.csv` (6,591 rows)
| Column | Type | Description |
| :--- | :--- | :--- |
| `account_id` | `string` (FK) | Unique account identifier. |
| `device_id` | `string` | Resolved physical device fingerprint identifier (`DEV_xxxxxx`). |
| `first_seen_date` | `timestamp` | First historical observation of this account using this physical device. |

#### `resolved_account_payment.csv` (9,601 rows)
| Column | Type | Description |
| :--- | :--- | :--- |
| `account_id` | `string` (FK) | Unique account identifier. |
| `payment_id` | `string` | Resolved payment instrument hash (card fingerprint, VPA hash, bank account). |
| `first_seen_date` | `timestamp` | First timestamp this payment instrument was charged or authorized. |

#### `resolved_account_address.csv` (9,681 rows)
| Column | Type | Description |
| :--- | :--- | :--- |
| `account_id` | `string` (FK) | Unique account identifier. |
| `address_id` | `string` | Normalized and geocoded physical delivery location identifier (`ADDR_xxxxxx`). |
| `first_seen_date` | `timestamp` | First delivery order placed using this delivery destination. |

#### `resolved_account_ip.csv` (6,594 rows)
| Column | Type | Description |
| :--- | :--- | :--- |
| `account_id` | `string` (FK) | Unique account identifier. |
| `ip_id` | `string` | Resolved IP address / subnet cluster identifier (`IP_xxxxxx`). |
| `first_seen_date` | `timestamp` | First observed session connection from this IP subnet. |

---

### 1.3 `orders.csv` (18,966 rows)
Historical transaction log recording e-commerce purchases, promo claims, and cashouts across the population.

| Column | Type | Description |
| :--- | :--- | :--- |
| `order_id` | `string` (PK) | Unique transaction identifier (`ORD_xxxxxx`). |
| `account_id` | `string` (FK) | Account placing the order. |
| `amount` | `float` | Order value in INR (`Rs.`). Mean order value = `Rs. 888.29`. |
| `timestamp` | `timestamp` | Exact transaction clearance timestamp (`YYYY-MM-DD HH:MM:SS`). |
| `product` | `string` | Product / voucher category purchased. |

---

### 1.4 `referrals.csv` (2,240 rows)
Directed graph of referral connections capturing both organic user invitations and orchestrated syndicate referral rings.

| Column | Type | Description |
| :--- | :--- | :--- |
| `referrer_id` | `string` (FK) | Account issuing the invitation / referral link. |
| `referred_id` | `string` (FK) | New account activated using the invitation code. |
| `referral_date` | `timestamp` | Timestamp when the referral link was claimed. |
| `bonus_amount` | `float` | Promotional incentive / cash credit awarded (e.g. `Rs. 50`–`Rs. 200`). |

---

### 1.5 `ground_truth.csv` (6,558 rows)
**Evaluation Only — Never ingested during graph construction or feature extraction.**

| Column | Type | Description |
| :--- | :--- | :--- |
| `account_id` | `string` (PK) | Account identifier matching `accounts.csv`. |
| `ring_id` | `string` (nullable) | Identifier of resource-sharing fraud ring (`RING_xxx`), or null. |
| `is_ring_member` | `bool` | `True` if the account belongs to a resource-sharing syndicate (484 accounts). |
| `coincidental_group_id` | `string` (nullable) | Identifier of benign sharing group (`COINC_xxxx`), or null. |
| `referral_ring_id` | `string` (nullable) | Identifier of referral-chain abuse ring (`REF_RING_xxx`), or null. |
| `is_referral_ring_member` | `bool` | `True` if the account belongs to a referral-chain fraud ring (74 accounts). |

---

### 1.6 `referral_ground_truth.csv` (2,240 rows)
**Evaluation Only.** Edge-level companion to `ground_truth.csv`'s account-level referral labels — one row per referral link in `referrals.csv`.

| Column | Type | Description |
| :--- | :--- | :--- |
| `referrer_id` | `string` (FK) | Account issuing the referral (matches `referrals.csv`). |
| `referred_id` | `string` (FK) | Account activated by the referral (matches `referrals.csv`). |
| `is_ring_referral` | `bool` | `True` if this specific referral edge is part of a referral-chain fraud ring, as opposed to an organic invitation. |

---

### 1.7 Synthetic Artifacts & Bridge Logs
* **`raw_to_true_resource.csv`** (32,467 rows): Cross-walk table recording raw noisy entity inputs vs. ground-truth resolved entities (e.g. typos, proxy shifts).
* **`bridge_log.csv`** (7 rows): Adversarial edge cases where a coincidental innocent bystander is connected via a single resource link to an active fraud ring, testing community detection robustness.

| Column | Type | Description |
| :--- | :--- | :--- |
| `type` | `string` | Bridge scenario type. |
| `a`, `b` | `string` | The two accounts joined by the bridging resource. |
| `resource_type` | `string` | Which entity type (`device`, `payment`, `address`, `ip`) forms the bridge. |
| `bridged_account` | `string` | The bystander account being bridged into the fraud cluster's orbit. |
| `ring` | `string` | The fraud ring the bystander is adversarially connected to. |
| `group` | `string` | The bystander's true (benign) coincidental group. |

---

## 2. Generated Pipeline & Evaluation Artifacts

### 2.1 `clusters.csv` (668 monitored accounts)
Output of Louvain community detection run on the multi-entity projection graph $G(V, E)$.
* `account_id`: Monitored account ID.
* `cluster_id`: Louvain community integer index ($0$ to $69$).
* `cluster_size`: Total count of member accounts residing inside this cluster.

---

### 2.2 `cluster_features.csv` & `cluster_predictions.csv` (70 Candidate Clusters)
Cluster-level feature matrix containing 17 engineered structural, temporal, and referral features (plus the `cluster_id` and `cluster_size` key/sizing columns):

#### Core Graph Topology Features:
* `cluster_size`: Number of distinct accounts in the community.
* `entity_reuse_ratio` ($\text{ERR}$): Degree of entity reuse across cluster members, computed as $1 - \frac{\text{distinct\_resources}}{\text{total\_usages}}$ across all linked devices, payment instruments, delivery addresses, and IP subnets. A value of $0.0$ indicates zero entity sharing (every usage is a unique entity), while values approaching $1.0$ indicate intense entity recycling across accounts.
* `internal_density`: Edge density within the community subgraph ($\frac{2E}{N(N-1)}$).
* `mean_degree_centrality`, `max_degree_centrality`: Distribution of node degrees within the cluster.
* `mean_pagerank`, `max_pagerank`: Structural influence and connection authority of member accounts.
* `mean_betweenness_centrality`, `max_betweenness_centrality`: Bridge and broker centrality scores.

#### Temporal & Transactional Features:
* `creation_span_days`: Days between the first and last member registration in the cluster.
* `creation_std_days`: Standard deviation of member registration timestamps (burstiness).
* `avg_order_amount`: Average transaction amount across all member purchases.
* `order_amount_cv`: Coefficient of variation of order amounts ($\frac{\sigma}{\mu}$).

#### Directed Referral Network Features (`REFERRAL_FEATURE_COLS`):
* `referral_cycle_ratio`: Fraction of cluster member accounts involved in directed referral cycles (accounts residing in strongly connected components with $\ge 2$ members, divided by cluster size $N$). Natural referral trees form directed acyclic graphs (DAGs); circular loops ($A \to B \to C \to A$) indicate bonus extraction collusion.
* `referral_resource_overlap_ratio`: Fraction of referral edges touching any cluster member that stay fully internal to the cluster ($\frac{\text{internal referral edges}}{\text{total referral edges involving cluster members}}$). Measures whether referral links are contained within the resource-sharing community rather than radiating outward to organic users.
* `median_referral_activation_days`: Median elapsed days between referral timestamp and referred account first order ($\text{first\_order\_date} - \text{referral\_date}$ across accounts referred by cluster members). Organic referrals exhibit natural human activation lag, whereas fraud rings activate immediately (defaults to $60.0$ days if no activation orders exist).
* `within_cluster_referral_density`: Fraction of unordered member account pairs connected by a referral link in either direction ($\frac{\text{unique connected pairs}}{\binom{N}{2}}$). Measures referral graph interconnectedness within the cluster.

#### Prediction Columns (`cluster_predictions.csv`):
* `y_true_is_ring`: Ground-truth label ($1$ if cluster contains true fraud ring members, $0$ if coincidental/benign).
* `oof_prob_logreg_pure_graph_referral`: **Champion Model** out-of-fold probability calibrated via 5-fold Stratified CV.
* Ablation probability baselines: `oof_prob_logreg_full`, `oof_prob_xgb_full`, `oof_prob_logreg_structural`, `oof_prob_logreg_pure_graph`.

---

### 2.3 `account_scores.csv` (488 rows) & `referral_cluster_features.csv` (8 rows)
* **`account_scores.csv`**: Intra-cluster account-level risk ranking, output by `account_scoring.py`. One row per account belonging to a flagged cluster, used by the cockpit to distinguish mastermind hubs from peripheral sleepers.
  * `account_id`, `cluster_id`: Identifiers.
  * `n_shared_resources`, `within_cluster_degree`, `within_cluster_edge_weight_sum`: Local connectivity within the cluster.
  * `creation_date_centrality`, `order_amount_centrality`: Normalized standing relative to other cluster members.
  * `account_risk_score`: Composite per-account ranking score.
* **`referral_cluster_features.csv`**: Referral-only feature slice (`referral_cycle_ratio`, `referral_resource_overlap_ratio`, `median_referral_activation_days`, `within_cluster_referral_density`) for the 8 clusters identified as referral rings.

---

### 2.4 `final_model.joblib`
Serialized production Champion Model (`sklearn.pipeline.Pipeline`):
* **Scaler**: `StandardScaler()` fit across training features.
* **Classifier**: Calibrated `LogisticRegression(max_iter=1000, random_state=42)`.
* **Feature Schema**: 7 Champion Features (`cluster_size`, `entity_reuse_ratio`, `internal_density`, `referral_cycle_ratio`, `referral_resource_overlap_ratio`, `median_referral_activation_days`, `within_cluster_referral_density`).

---

### 2.5 Benchmark & Temporal Audit Artifacts
* **`phase3_detection_latency_audit.csv`** (28 rows) & **`phase3_counterfactual_summary.json`**: The canonical temporal audit output, produced by `phase3_temporal_backtest.py` (101 snapshots at 7-day intervals). Includes the resource-vs-referral subgroup breakdown (e.g. `resource_sharing_median_latency_days`, `referral_chain_median_latency_days`).
* **`temporal_detection_latencies.csv`** (28 rows) & **`temporal_backtest_summary.json`**: Compatibility-format copies of the same run, written by `phase3_temporal_backtest.py` under the filenames the older `temporal_reconstruction.py` module used, so downstream readers don't need to change.
* **`pooled_threshold_selection_results.csv`** & **`pooled_threshold_selection_summary.json`**: 15-seed pooled sensitivity analysis results establishing the plateau midpoint threshold (`0.1333`).
* **`threshold_sweep_results.csv`**: 0.1x–100x false-positive-cost-multiplier sweep, per candidate model.
* **`bootstrap_threshold_ci_results.csv`**: Per-resample detail underlying the 10,000-resample bootstrap confidence intervals.
* **`naive_baseline_results.csv`**: Precision/recall/F1 of the naive shared-address-count baseline (`naive_baseline.py`) across several size thresholds, for comparison against the Champion model.
* **`metrics_summary.json` & `final_report.md`**: Official audit report capturing the verified cluster-level confusion matrix ($\text{TP}=26, \text{FP}=0, \text{FN}=0, \text{TN}=44$ — 26 of the 70 candidate clusters), protected value (`Rs. 708,255`), and review cost (`Rs. 1,777`).