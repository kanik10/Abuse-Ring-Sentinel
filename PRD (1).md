# PRD: Abuse-Ring Sentinel

**Track:** Razorpay AI Buildathon 2026 — Track 02, AI Risk Manager
**Loss class:** coordinated abuse rings (promo/referral bonus farming)
**Window:** Aug 29 – Sep 5, 2026
**Status:** Day 1 (synthetic data) complete

---

## 1. Overview

A defense-only detection system that identifies coordinated groups of accounts
("rings") colluding to exploit promo or referral bonuses by sharing devices,
payment instruments, or delivery addresses across many nominally-separate
accounts. The system outputs a risk score and supporting evidence per
detected cluster for human review — it never takes an automated action on
an account.

## 2. Problem Statement

Individually, each account in an abuse ring can look unremarkable — a normal
signup, a normal first order. The loss is only visible at the *relationship*
level: the same device claiming five signup bonuses, the same address
receiving fifteen "first orders." Row-by-row fraud scoring structurally
cannot see this pattern; it requires modeling accounts as a graph and finding
densely-connected clusters. This is the loss class Track 02 names directly —
"fraud... quietly eat[ing] margin" — and the track's own example directions
list it as "Abuse-ring sentinel."

## 3. Goals

- **G1 — Detect:** find coordinated rings in a population of accounts using
  shared-resource graph structure.
- **G2 — Score honestly:** report precision, recall, and F1 on a held-out
  set of rings never seen during training, plus an explicit false-positive
  cost analysis.
- **G3 — Explain:** every flagged cluster comes with the specific evidence
  (which resources, which features) that triggered it.
- **G4 — Stay defense-only:** the system's only possible output is a
  flag-for-review signal. No code path exists that blocks, freezes, or
  cancels anything automatically.

## 4. Non-Goals

- Not building a GNN-based model (PyTorch Geometric) — stretch goal only if
  the core pipeline finishes early.
- Not integrating with live payment rails or real merchant data.
- Not attempting real-time/streaming detection — this is a batch
  detector over a snapshot population.
- Not auto-remediating anything. See G4.

## 5. Users

- **Primary:** a fraud/risk analyst who receives a queue of flagged clusters
  with evidence and decides whether to investigate further.
- **Secondary:** the hackathon panel evaluating the submission against
  "the bar" — honest metrics including false-positive cost, strictly
  defense-only.

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Generate a synthetic population of accounts: independent "legit" accounts, accounts organized into camouflaged abuse rings, and legit accounts with coincidental (benign) resource overlap |
| FR2 | Construct an account-level graph from shared devices/payment instruments/addresses, with edges weighted by inverse resource frequency |
| FR3 | Run community detection (Louvain) to produce candidate clusters |
| FR4 | Engineer cluster-level features: size, entity-reuse ratio, internal density, creation-time burstiness, order-value dispersion, resource commonness |
| FR5 | Train a classifier (XGBoost/LightGBM) on cluster features to produce a calibrated risk score per cluster, split by ring ID to prevent leakage |
| FR6 | Select an operating threshold using an explicit false-positive cost table, not F1 alone |
| FR7 | Output schema is exactly `{cluster_id, risk_score, evidence, recommended_action: "flag_for_review"}` — no other action type is representable |
| FR8 | Log an audit trail per flagged cluster: which resources, which feature values, at what score |
| FR9 | Report precision/recall/F1/PR-AUC on the held-out ring split, plus the false-positive cost table |
| FR10 (optional) | Minimal demo (Streamlit or notebook walkthrough) showing a flagged cluster and its evidence |

## 7. Non-Functional Requirements

- **Reproducibility:** synthetic data generation is seeded; same seed
  produces the same population.
- **Auditability:** every score must be traceable to specific evidence, not
  a black-box number.
- **Structural safety:** defense-only is enforced by the output schema, not
  by a comment or a README promise.
- **Runnable from README alone:** a stranger cloning the repo can reproduce
  results without asking you anything.

## 8. System Architecture

```
raw entities (accounts, devices, payments, addresses, orders)
        │
        ▼
graph construction (bipartite account–resource → weighted account–account)
        │
        ▼
community detection (Louvain)
        │
        ▼
cluster feature engineering
        │
        ▼
classifier → calibrated risk score
        │
        ▼
threshold selection (cost-based)
        │
        ▼
human review queue  ← (only output; no automated action)
```

### Tech stack

| Layer | Tool | Cost |
|---|---|---|
| Data generation | pandas, numpy, Faker | Free |
| Graph construction | NetworkX | Free |
| Community detection | python-louvain | Free |
| Classifier | XGBoost or LightGBM | Free |
| Evaluation | scikit-learn | Free |
| Demo (optional) | Streamlit | Free |
| Compute | Local / Google Colab / Kaggle Notebooks | Free |

## 9. Data Requirements

See `DATA_DICTIONARY.md` for the full schema. Key constraint: `ground_truth.csv`
(`ring_id`, `is_ring_member`, `coincidental_group_id`) is used **only** for
evaluation. It must never be joined into the graph-construction or
feature-engineering steps — doing so invalidates every downstream metric.

Current synthetic population (seed=42): 6,414 accounts — 6,000 legit,
414 across 20 rings (sizes 7–39), plus 147 accounts in 50 coincidental
benign-overlap groups.

## 10. Success Metrics

- **Precision / Recall / F1** on rings held out by ring ID (never split by
  row — that leaks).
- **PR-AUC** across thresholds.
- **False-positive cost table:** cost of reviewing a flagged legit cluster
  vs. cost of a missed ring (avg. fraudulent value × ring size), used to
  justify the chosen threshold.
- **Qualitative check:** do false positives concentrate on the coincidental
  overlap groups (expected, defensible) or scatter randomly (a sign the
  model isn't learning the right signal)?

## 11. Defense-Only Compliance

The system:
- **Never** blocks, cancels, freezes, or rate-limits an account.
- **Never** takes an action without a human in the loop.
- **Always** attaches evidence to a flag, so a reviewer can override it.

This directly satisfies the track's disqualification condition
("strictly defense-only: anything offense-capable is disqualified") —
there is no code path that could be repurposed to *cause* harm to an
account, only to surface it for review.

## 12. Milestones

| Day | Date | Deliverable | Status |
|---|---|---|---|
| 1 | Aug 29 | Synthetic data + ground truth | ✅ Done |
| 2 | Aug 30 | Graph + Louvain, first precision/recall number | Pending |
| 3 | Aug 31 | Cluster features + trained classifier | Pending |
| 4 | Sep 1 | Tuned metrics + false-positive cost table | Pending |
| 5 | Sep 2 | Defense-only output schema + demo | Pending |
| 6 | Sep 3 | Repo hygiene, README, architecture diagram | Pending |
| 7 | Sep 4 | Pitch video | Pending |
| — | Sep 5 | Submission | Pending |

## 13. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Louvain doesn't cleanly separate rings from noise | Resolution-parameter grid search; inverse-frequency edge weighting |
| Synthetic data too easy (trivially separable) or too hard (no signal) | Coincidental overlap groups + camouflaged resource rotation, tuned by inspecting degree distributions (done in Day 1) |
| Train/test leakage inflates metrics | Split strictly by ring ID, never by row |
| Timeline slip on the hardest days (2–3) | Day 2's checkpoint (a working, even bad, precision/recall number) is non-negotiable before continuing |
| Defense-only framing treated as a README afterthought | Output schema enforces it structurally from Day 5, reviewed again during Day 6 repo cleanup |

## 14. Deliverables (per track requirements)

- Public GitHub repository (code, data generation script, README)
- 5-minute pitch video
- Architecture diagram

## 15. Open Questions / Future Work

- Would a GNN (GraphSAGE-style) meaningfully outperform Louvain + gradient
  boosting on this data, given more time?
- How would this generalize to a streaming/real-time setting rather than a
  batch snapshot?
- What would the actual cost inputs (support ticket cost, average
  fraudulent bonus value) be if grounded in real Razorpay merchant data
  rather than assumed figures?
