# Statistical Analysis for Manuscript Rebuttal

## Purpose

This project addresses reviewer comment Section 6.1, p2 on a manuscript comparing 5 reconstruction methods for accelerated MRI. The reviewer requested statistical significance testing, effect sizes, and 95% CIs for two specific claims:

- **Claim 1**: "Our full method outperforms the other two diffusion-based methods across all anatomies, metrics and accelerations."
- **Claim 2**: "Better SSIM than end-end methods except PD knee data."

The deliverable is a set of tables and a prose summary suitable for inclusion in the rebuttal letter.

## Methods being compared

Five methods, named consistently throughout all outputs:

- `modl+loupe` — MoDL backbone with LOUPE sampling pattern
- `modl+poisson` — MoDL backbone with Poisson-disc sampling
- `diffusion+loupe` — diffusion backbone with LOUPE sampling
- `diffusion+poisson` — diffusion backbone with Poisson-disc sampling
- `ours` — diffusion backbone with the proposed (DOGG) sampling pattern

Categories:
- **End-to-end methods** (for Claim 2): `modl+loupe`, `modl+poisson`
- **Other diffusion methods** (for Claim 1): `diffusion+loupe`, `diffusion+poisson`

## Anatomies and accelerations

- 4 anatomies: `t2_brain`, `pd_knee`, `pdfs_knee`, `all_knee`
- Accelerations: R4, R8, R12, R16, R20 (verified present in every CSV)
- 3 metrics: SSIM, PSNR, NRMSE

Direction:
- Higher is better: SSIM, PSNR
- Lower is better: NRMSE

**Note on `all_knee`**: `combined_results.csv` contains 551 unique slice IDs which equals `pd_results.csv` (281) + `pdfs_results.csv` (270). It is the union of the two knee subtypes, not an independent third dataset. The rebuttal summary should make this clear so the reviewer doesn't double-count.

## Input data layout

8 CSV files in `data/`, paired by anatomy. Each anatomy's two files report **the same set of test slices** evaluated by different methods, so pairing extends across the file boundary.

| anatomy   | diffusion file        | modl file                | n_slices |
|-----------|-----------------------|--------------------------|----------|
| t2_brain  | brain_results.csv     | MoDL_brain_results.csv   | 280      |
| pd_knee   | pd_results.csv        | MoDL_pd_results.csv      | 281      |
| pdfs_knee | pdfs_results.csv      | MoDL_pdfs_results.csv    | 270      |
| all_knee  | combined_results.csv  | MoDL_combined_results.csv| 551      |

### CSV schema

Each row is one (method × acceleration) result. Relevant columns:

- `recon` — backbone: `diffusion` or `MoDL` (note the capitalization in the modl files; the pipeline must match case-insensitively).
- `pattern` — sampling pattern: `LOUPE`, `Poisson`, or `DOGG`
- `acceleration` — integer R factor
- `anatomy` — anatomy string from the file (sanity-check matches the filename mapping)
- `name` — full row identifier including K/BS for DOGG rows (e.g., `brain_DOGG_R12_K12_BS12`, `PD_DOGG_R8_K16_BS8`). Used to extract DOGG hyperparameters for logging.
- `mean ssim`, `std ssim`, `mean psnr`, `std psnr`, `mean nrmse`, `std nrmse` — summary stats. Used **only** for the "ours" row selection (see below); not used for any statistical test.
- Per-slice columns named `<file_id> <metric>` where metric ∈ {`ssim`, `psnr`, `nrmse`}. Slice IDs vary by anatomy: brain uses `file_brain_AXT2_<...>_slice_<n>`, knee uses `file<digits>_slice_<n>`. The pipeline must extract slice IDs by **stripping the trailing metric token only** (do not assume any particular prefix).

### Method derivation from `recon` + `pattern`

| recon (case-insensitive) | pattern | method            |
|--------------------------|---------|-------------------|
| modl                     | LOUPE   | modl+loupe        |
| modl                     | Poisson | modl+poisson      |
| diffusion                | LOUPE   | diffusion+loupe   |
| diffusion                | Poisson | diffusion+poisson |
| diffusion                | DOGG    | ours (see below)  |

### "Ours" selection

Each diffusion CSV contains 4–5 DOGG rows per acceleration with different K and BS hyperparameters. For each (anatomy, acceleration), "ours" is the DOGG row with the lowest `mean nrmse`. This selection rule is stated in the manuscript, so it is not cherry-picking.

The pipeline must:

1. Group DOGG rows within an (anatomy, acceleration).
2. Pick the row with `idxmin()` on `mean nrmse`.
3. Log the chosen K/BS (parsed from the `name` column) and its mean NRMSE to `results/diagnostics.log`, alongside the K/BS and NRMSE of the rows it beat. This makes the selection fully auditable.
4. If `mean nrmse` is missing or NaN, halt with an error rather than guessing.

## Statistical setup

### Pairing

Slices are **paired across methods, including across the diffusion/modl file boundary**. For a given (anatomy, acceleration), the same set of test slices was reconstructed by every method, so per-slice scores form 5-way matched tuples (3 methods from the diffusion file + 2 from the modl file). The pipeline must:

1. Extract the slice IDs (the `<file_id>` portion of each per-slice column) per row.
2. Verify the slice ID set is identical across all 5 method rows for that (anatomy, acceleration). Empirically this holds for the supplied data, but the check must run on every load in case future runs use different data.
3. If sets differ, take the intersection, log a warning naming the dropped slices, and proceed on the intersection.

### Tests

For every comparison (ours vs one competitor) at each (anatomy, acceleration, metric):

- **Primary**: paired Wilcoxon signed-rank test, **two-sided**.
- **Secondary**: paired t-test, **two-sided** (cross-check; Wilcoxon is the headline test).
- **Direction flag**: alongside each two-sided p, report the sign of the median paired difference (ours − competitor), oriented so positive always means "ours is better" (i.e., flip sign for NRMSE). Column should be one of `ours_better`, `ours_worse`, `tie`.

### Effect sizes and CIs

For each comparison, report:

- **Median paired difference** (ours − competitor, sign-flipped for NRMSE so positive = ours better), with **95% BCa bootstrap CI** using 10,000 resamples and `seed: 42`.
- **Paired Cohen's d** = mean(diff) / sd(diff), with 95% CI (use the standard parametric CI; document the formula in code comments).
- **Matched-pairs rank-biserial correlation** for the Wilcoxon (the natural Wilcoxon effect size).
- For SSIM specifically, also report the median diff in **percentage points** (×100) for readability.

### Multiple-comparisons correction

**None.** Report raw p-values only. The manuscript's claims are descriptive ("outperforms across all"), so the audience cares about the per-cell pattern more than family-wise error.

## Comparison families

### Claim 1 family (ours vs other diffusion)

For every (anatomy, acceleration ∈ {4, 8, 12, 16, 20}, metric ∈ {SSIM, PSNR, NRMSE}, competitor ∈ {diffusion+loupe, diffusion+poisson}):

- Run the test pair above.
- The claim is supported at that cell iff: ours is directionally better **and** Wilcoxon p < 0.05.
- Cells where ours is directionally worse must be flagged prominently — they directly contradict the claim and must be visible before submitting the rebuttal.

### Claim 2 family (ours vs end-to-end, SSIM only)

For every (anatomy, acceleration, competitor ∈ {modl+loupe, modl+poisson}), metric = SSIM only:

- Same test pair.
- The reviewer's prediction: ours is better everywhere except PD knee. Report PD knee separately and check whether the data agree.

## Outputs

Write everything to `results/`:

- `results/long_data.parquet` — the melted (anatomy, method, acceleration, slice_id, metric, value) table after the slice-intersection step. Useful for re-analysis.
- `results/ours_selection.csv` — one row per (anatomy, acceleration) recording which DOGG K/BS was chosen, its mean NRMSE, and the alternatives that lost. The auditable record of the "ours" choice.
- `results/claim1_table.csv` — one row per (anatomy, acceleration, metric, competitor) with: n_slices, mean_ours, mean_competitor, median_diff, ci95_low, ci95_high, cohens_d, d_ci_low, d_ci_high, rank_biserial, wilcoxon_p, ttest_p, direction.
- `results/claim2_table.csv` — same columns, restricted to SSIM and end-to-end competitors.
- `results/claim1_summary.md` — prose for the rebuttal: one paragraph per anatomy summarizing how many cells support the claim, listing any contradicting cells with their numbers.
- `results/claim2_summary.md` — same for Claim 2, with explicit handling of the PD-knee exception.
- `results/diagnostics.log` — slice-set mismatches, missing accelerations, "ours" selections, and any other warnings.

## Pipeline

Implement as separate scripts so each can be re-run independently:

1. `src/load_data.py` — read all 8 CSVs, build the long table, save to parquet.
2. `src/select_ours.py` — for each (anatomy, acceleration), pick the DOGG row with the lowest `mean nrmse`, drop the unused DOGG rows, write `results/ours_selection.csv`.
3. `src/run_tests.py` — produce both claim tables.
4. `src/make_summaries.py` — produce both markdown summaries.
5. `Makefile` with `make all` running the pipeline end to end and `make clean` removing `results/`.

## Dependencies

`requirements.txt` should pin: `numpy`, `pandas`, `scipy`, `statsmodels`, `pyarrow`. Use `scipy.stats.wilcoxon`, `scipy.stats.ttest_rel`, and `scipy.stats.bootstrap` with `method='BCa'` for the CI on the median paired difference.

## Reproducibility

- `seed: 42` in `config.yaml` for the bootstrap.
- All scripts must be deterministic given the seed.
- `requirements.txt` pinned to specific versions.

## Quality checks the pipeline must perform

Before producing tables, verify and log:

- Each anatomy has exactly one diffusion CSV and one modl CSV at the paths in `config.yaml`.
- The 5 methods are all present at every acceleration after "ours" selection, or log which are missing.
- Slice ID sets match across methods within an (anatomy, acceleration) — log mismatches.
- For each test, the sign of the median paired difference matches the sign of the mean difference (sanity check on direction).
- n_slices ≥ 5 for any test to run (Wilcoxon is unreliable below this); skip and log otherwise.
- Report `n_slices` in every output row so the reviewer can see the sample size.

## Style for the rebuttal summaries

Write the markdown summaries in the voice of the manuscript authors responding to the reviewer. Quote the original claim, then state what the data show, then table-reference the supporting numbers. Be honest about any cells that contradict the claim — the reviewer will check.
