# Day 4 Final Report — Abuse-Ring Sentinel Operating Point

## Chosen model and threshold
- **Model:** Logistic Regression (pure graph)
- **Threshold:** 0.3529693519
- **Why:** cost-optimal threshold was stable across a 0.1x-100x sweep of
  false-positive-cost assumptions (Day 4 Phase 2) -- more robust than the
  XGBoost alternative, which shifted its recommendation at extreme (100x)
  assumptions. Also the more interpretable model, and the same feature set
  (pure graph topology, no timing or order-amount features) already
  recommended in Day 3 as the more generalizable result.

## Cluster-level confusion matrix (out-of-fold, 70 clusters)
| | Predicted: ring | Predicted: not ring |
|---|---|---|
| **Actual: ring** | TP = 18 | FN = 0 |
| **Actual: not ring** | FP = 0 | TN = 52 |

- Precision: 1.000
- Recall: 1.000

## Account-level cost/benefit at this threshold
- Ring fraud value protected (caught): Rs.411,887.55
- Ring fraud value still missed: Rs.0.00
- Legitimate accounts wrongly caught in a flagged cluster: 3 total -- 3 coincidental/benign-lookalike (100%), 0 other (0%)
- Cost of those false positives, at 1x avg order value per account: Rs.2,675.90
- **Net: protects Rs.411,888 of fraud at a cost of roughly
  Rs.2,676 in false-positive review/friction (at a conservative
  1x-avg-order-value cost assumption) -- a ~154x return.**

## Honest limitations of this number
- N=70 clusters (18 positive at this threshold's flagging count) -- treat
  precision/recall as directionally reliable, not statistically tight.
- The false-positive cost assumption (1x avg order value per wrongly-
  flagged account) is a modeling choice, not a measured business figure --
  see Day 4 Phase 1 for why it's swept rather than asserted as fact.
- This threshold was tuned on the SAME synthetic dataset it's evaluated on
  (out-of-fold within that one dataset, not a separate holdout population).
  A genuinely held-out second synthetic population, or real data, would be
  needed before trusting this threshold in production.
