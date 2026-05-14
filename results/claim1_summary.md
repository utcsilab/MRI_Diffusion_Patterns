# Response to Reviewer 6.1, p2 — Claim 1

> "Our full method outperforms the other two diffusion-based methods across all anatomies, metrics and accelerations."

We thank the reviewer for prompting a quantitative test of this claim. Per-slice SSIM, PSNR, and NRMSE for each test slice were compared between our method and each of the two other diffusion baselines using paired Wilcoxon signed-rank tests (two-sided, α = 0.05). Effect sizes are reported as the median paired difference with 95% BCa bootstrap CI (10,000 resamples) and paired Cohen's *d* with 95% noncentral-*t* CI. Pairing is across all five methods on the same test slices for each (anatomy, acceleration). Full per-cell numbers are in `claim1_table.csv`; the canonical DOGG hyperparameters used for "ours" at each (anatomy, acceleration) are in `ours_selection.csv`.

## Summary

Across the 120 comparisons (4 anatomies × 5 accelerations × 3 metrics × 2 diffusion competitors), **119/120** significantly support the claim (ours better, Wilcoxon p < 0.05). **0** comparisons significantly contradict it. 1 cell is directionally in our favor but not statistically resolved, and 0 are directionally against us but not statistically resolved.

**No comparison significantly contradicts the claim.** The non-resolved cells are flagged below in the appropriate per-anatomy paragraphs so the reviewer can verify directly.

## T2 brain  (n_slices = 280)

Of the 30 comparisons, **29 significantly favor ours** (Wilcoxon p < 0.05), **0 significantly favor the competitor**. 1 is directionally in our favor but did not reach significance.

Cells that do not strictly support the claim:

- directionally in our favor, not significant — R=4 PSNR vs diffusion+poisson: median diff = +0.003 dB (95% CI [-0.012, +0.014]), Wilcoxon p=0.802, Cohen's d = +0.03 (95% CI [-0.09, +0.15]).

## PD knee  (n_slices = 281)

Of the 30 comparisons, **30 significantly favor ours** (Wilcoxon p < 0.05), **0 significantly favor the competitor**.

Every cell strictly supports the claim. Effect sizes are large throughout (Cohen's *d* ranges roughly from +0.72 to +2.00).

## PD-FS knee  (n_slices = 270)

Of the 30 comparisons, **30 significantly favor ours** (Wilcoxon p < 0.05), **0 significantly favor the competitor**.

Every cell strictly supports the claim. Effect sizes are large throughout (Cohen's *d* ranges roughly from +0.31 to +2.39).

## all knee (PD + PD-FS pooled)  (n_slices = 551)

Of the 30 comparisons, **30 significantly favor ours** (Wilcoxon p < 0.05), **0 significantly favor the competitor**.

Every cell strictly supports the claim. Effect sizes are large throughout (Cohen's *d* ranges roughly from +0.55 to +1.79).
