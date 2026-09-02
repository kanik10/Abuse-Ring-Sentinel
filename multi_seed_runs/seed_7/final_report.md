# Day 4 Final Report — Abuse-Ring Sentinel Operating Point

## Chosen model and threshold
- **Model:** Logistic Regression (pure graph)
- **Threshold:** 0.48111024428768
- **Why:** cost-optimal threshold was stable across the full 0.1x-100x
  false-positive-cost sweep (Day 4 Phase 2). The model is also
  interpretable, and uses the same pure graph-topology feature set already
  recommended in Day 3 as the more generalizable result.

## Cluster-level confusion matrix (out-of-fold, 64 clusters)
| | Predicted: ring | Predicted: not ring |
|---|---|---|
| **Actual: ring** | TP = 20 | FN = 0 |
| **Actual: not ring** | FP = 0 | TN = 44 |

- Precision: 1.000
- Recall: 1.000

## Account-level cost/benefit at this threshold
- Ring fraud value protected (caught): Rs.534,240.31
- Ring fraud value still missed: Rs.0.00
- Legitimate accounts wrongly caught in a flagged cluster: 15 total -- 15 coincidental/benign-lookalike (100%), 0 other (0%)
- Cost of those false positives, at 1x avg order value per account: Rs.13,373.34
- **Net: protects Rs.534,240 of fraud at a cost of roughly
  Rs.13,373 in false-positive review/friction (at a conservative
  1x-avg-order-value cost assumption) -- a ~40x return.**

## Honest limitations of this number
- N=64 clusters (20 flagged at this threshold) -- treat
  precision/recall as directionally reliable, not statistically tight.
- The false-positive cost assumption (1x avg order value per wrongly-
  flagged account) is a modeling choice, not a measured business figure --
  see Day 4 Phase 1 for why it's swept rather than asserted as fact.
- This threshold was tuned on the SAME synthetic dataset it's evaluated on
  (out-of-fold within that one dataset, not a separate holdout population).
  A genuinely held-out second synthetic population, or real data, would be
  needed before trusting this threshold in production.

## Threshold stability (bootstrap, B=10000 resamples)
Nonparametric percentile bootstrap over clusters, resampled with
replacement. Answers a different question than the confusion matrix
above: not "how good is this threshold" but "how much would a different
draw of clusters have changed the answer."

- Precision at the locked threshold: 95% CI = [1.000, 1.000]
- Recall at the locked threshold: 95% CI = [1.000, 1.000]
- Total cost at the locked threshold: 95% CI = [Rs.3,566, Rs.25,855]
- Each resample's OWN cost-optimal threshold: min=0.8361, median=0.8361, max=0.9873
- 0.0% of resamples had their own optimum within
  +/-0.05 of the locked threshold (0.4811) -- read this as
  how confidently "single" this operating point really is, not as an
  error bar on precision/recall.
