# Abuse-Ring Sentinel — Complete Data Dictionary

**Dataset Version:** V2 (Multi-Entity Bipartite Resolution + Directed Referral Abuse Rings)  
**Location:** Canonical data resides in `day1_data/`  
**Population Size:** 6,558 Accounts | 18,966 Orders | 2,240 Referrals | 26 Ground-Truth Fraud Rings

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

### 1.6 Synthetic Artifacts & Bridge Logs
* **`raw_to_true_resource.csv`** (10,544 rows): Cross-walk table recording raw noisy entity inputs vs. ground-truth resolved entities (e.g. typos, proxy shifts).
* **`bridge_log.csv`** (10 rows): Adversarial edge cases where a coincidental innocent bystander is connected via a single resource link to an active fraud ring, testing community detection robustness.

---

## 2. Generated Pipeline & Evaluation Artifacts

### 2.1 `clusters.csv` (668 monitored accounts)
Output of Louvain community detection run on the multi-entity projection graph $G(V, E)$.
* `account_id`: Monitored account ID.
* `cluster_id`: Louvain community integer index ($0$ to $69$).
* `cluster_size`: Total count of member accounts residing inside this cluster.

---

### 2.2 `cluster_features.csv` & `cluster_predictions.csv` (70 Candidate Clusters)
Cluster-level feature matrix containing 18 engineered structural, temporal, and referral features:

#### Core Graph Topology Features:
* `cluster_size`: Number of distinct accounts in the community.
* `entity_reuse_ratio` ($\text{ERR}$): Number of unique shared physical entities divided by cluster size.
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
* `referral_cycle_ratio`: Fraction of referral connections forming directed closed loops (e.g. $A \to B \to C \to A$).
* `referral_resource_overlap_ratio`: Proportion of referral edges that also share a device, IP, or payment instrument.
* `median_referral_activation_days`: Median days between account creation and referral link redemption.
* `within_cluster_referral_density`: Ratio of internal referral edges to theoretically possible directed edges.

#### Prediction Columns (`cluster_predictions.csv`):
* `y_true_is_ring`: Ground-truth label ($1$ if cluster contains true fraud ring members, $0$ if coincidental/benign).
* `oof_prob_logreg_pure_graph_referral`: **Champion Model** out-of-fold probability calibrated via 5-fold Stratified CV.
* Ablation probability baselines: `oof_prob_logreg_full`, `oof_prob_xgb_full`, `oof_prob_logreg_structural`, `oof_prob_logreg_pure_graph`.

---

### 2.3 `final_model.joblib`
Serialized production Champion Model (`sklearn.pipeline.Pipeline`):
* **Scaler**: `StandardScaler()` fit across training features.
* **Classifier**: Calibrated `LogisticRegression(max_iter=1000, random_state=42)`.
* **Feature Schema**: 7 Champion Features (`cluster_size`, `entity_reuse_ratio`, `internal_density`, `referral_cycle_ratio`, `referral_resource_overlap_ratio`, `median_referral_activation_days`, `within_cluster_referral_density`).

---

### 2.4 Benchmark & Temporal Audit Artifacts
* **`temporal_detection_latencies.csv`** (28 rows): Backtest lifecycle table recording ring formation date, first flag date, detection latency in days, pre-flag volume, and prevented volume percentage across 101 historical snapshots.
* **`temporal_backtest_summary.json`**: Macro benchmark metrics (101 snapshots, 100% detection rate, 30-day median lag, 90.4% volume prevented).
* **`pooled_threshold_selection_summary.json`**: 15-seed pooled sensitivity analysis results establishing the plateau midpoint threshold (`0.1333`).
* **`metrics_summary.json` & `final_report.md`**: Official audit report capturing verified confusion matrix ($\text{TP}=26, \text{FP}=0, \text{FN}=0, \text{TN}=44$), protected value (`Rs. 708,255`), and review cost (`Rs. 1,777`).
