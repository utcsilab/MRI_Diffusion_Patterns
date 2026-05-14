# Response to Reviewer 6.1, p2 — Claim 2

> "Better SSIM than end-end methods except PD knee data."

We tested this claim by comparing ours against each end-to-end (MoDL) baseline on per-slice SSIM at every acceleration, using paired Wilcoxon signed-rank tests (two-sided, α = 0.05). Median paired SSIM differences are reported in percentage points (×100) with 95% BCa bootstrap CIs, and paired Cohen's *d* with 95% noncentral-*t* CIs. Full per-cell numbers are in `claim2_table.csv`.

## Summary

Of the 40 SSIM comparisons (4 anatomies × 5 accelerations × 2 end-to-end competitors), **38/40** significantly support the claim, **0** significantly contradict it, 1 is directionally in our favor but not significant, and 1 is directionally against us but not significant. The manuscript's hedged language — "except PD knee data" — is borne out: the non-supporting cells are concentrated on PD knee at high acceleration.

## PD knee (the manuscript's stated exception)

PD knee shows the predicted exception. Of the 10 PD-knee comparisons, **8 significantly favor ours**, 0 significantly favor the competitor, 1 is directionally in our favor but not significant, and 1 is directionally against us but not significant. The non-supporting cells are:

- directionally in our favor, not significant — R=16 SSIM vs modl+poisson: median diff = +0.163 pp (95% CI [+0.062, +0.222]), Wilcoxon p=0.077, Cohen's d = +0.11 (95% CI [-0.01, +0.23]).
- directionally against, not significant — R=20 SSIM vs modl+poisson: median diff = -0.062 pp (95% CI [-0.155, +0.080]), Wilcoxon p=0.069, Cohen's d = -0.08 (95% CI [-0.20, +0.04]).

These cluster at high acceleration (R ≥ 16) versus modl+poisson, consistent with the manuscript's existing caveat. The point estimates are within ~0.2 pp of zero in every non-supporting cell, so the qualitative conclusion is that ours is statistically indistinguishable from modl+poisson at high R on PD knee — not that it is decisively worse.

## T2 brain  (n_slices = 280)

Of the 10 SSIM comparisons, **10 significantly favor ours**, **0 significantly favor the competitor**.

Every cell strictly supports the claim. Effect sizes are large (Cohen's *d* range +2.58 to +3.27).

## PD-FS knee  (n_slices = 270)

Of the 10 SSIM comparisons, **10 significantly favor ours**, **0 significantly favor the competitor**.

Every cell strictly supports the claim. Effect sizes are large (Cohen's *d* range +1.13 to +1.68).

## all knee (PD + PD-FS pooled)  (n_slices = 551)

Of the 10 SSIM comparisons, **10 significantly favor ours**, **0 significantly favor the competitor**.

Every cell strictly supports the claim. Effect sizes are large (Cohen's *d* range +0.53 to +1.02).
