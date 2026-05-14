"""Run paired tests for Claim 1 and Claim 2 and write the result tables.

For each (anatomy, acceleration, metric, competitor):
  - paired Wilcoxon signed-rank, two-sided  (primary)
  - paired t-test, two-sided                (secondary)
  - median paired difference (oriented so positive == ours better;
    NRMSE flipped so smaller-is-better becomes positive-is-better),
    with 95% BCa bootstrap CI (10,000 resamples, seed 42; falls back
    to percentile CI if BCa is degenerate)
  - paired Cohen's d with 95% noncentral-t inversion CI
  - matched-pairs rank-biserial correlation
  - SSIM rows additionally report median diff in percentage points

Outputs: results/claim1_table.csv, results/claim2_table.csv.
Diagnostics appended to results/diagnostics.log.
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LONG_PARQUET = RESULTS / "long_data.parquet"
CLAIM1_CSV = RESULTS / "claim1_table.csv"
CLAIM2_CSV = RESULTS / "claim2_table.csv"
DIAG_LOG = RESULTS / "diagnostics.log"

OUTPUT_COLS = [
    "anatomy", "acceleration", "metric", "competitor",
    "n_slices", "mean_ours", "mean_competitor",
    "median_diff", "ci95_low", "ci95_high",
    "median_diff_pp", "ci95_low_pp", "ci95_high_pp",
    "cohens_d", "d_ci_low", "d_ci_high",
    "rank_biserial", "wilcoxon_p", "ttest_p", "direction",
]


def setup_logger() -> logging.Logger:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_tests")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(DIAG_LOG, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


def paired_d_ci(d: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for paired Cohen's d via noncentral-t inversion.

    Idea: t_obs = d*sqrt(n) follows nct(df=n-1, ncp=d_pop*sqrt(n)). Find
    ncp values whose nct distribution places t_obs at the 1 - alpha/2 and
    alpha/2 quantiles; divide by sqrt(n) to recover d.
    """
    if not np.isfinite(d) or n < 2:
        return float("nan"), float("nan")
    df = n - 1
    t_obs = d * math.sqrt(n)
    # nct.cdf(t_obs, df, ncp) is monotonically decreasing in ncp.
    span = max(50.0, 25.0 * (abs(t_obs) + 1.0))

    def f(ncp: float, target: float) -> float:
        return stats.nct.cdf(t_obs, df, ncp) - target

    # nct.cdf can emit a benign RuntimeWarning at very extreme ncp values during
    # bracketing, even when brentq converges to a valid root. Mute locally.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in _nct_cdf",
                                    category=RuntimeWarning)
            ncp_low = brentq(f, t_obs - span, t_obs + span, args=(1 - alpha / 2,))
            ncp_high = brentq(f, t_obs - span, t_obs + span, args=(alpha / 2,))
    except ValueError:
        return float("nan"), float("nan")
    return ncp_low / math.sqrt(n), ncp_high / math.sqrt(n)


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial for paired Wilcoxon (zeros dropped)."""
    nz = diff[diff != 0]
    if nz.size == 0:
        return float("nan")
    abs_ranks = stats.rankdata(np.abs(nz))
    w_pos = abs_ranks[nz > 0].sum()
    w_neg = abs_ranks[nz < 0].sum()
    total = w_pos + w_neg
    if total == 0:
        return float("nan")
    return float((w_pos - w_neg) / total)


def median_ci(diff: np.ndarray, log: logging.Logger, label: str,
              n_resamples: int = 10000, seed: int = 42) -> tuple[float, float, str]:
    """BCa bootstrap CI for the median; falls back to percentile if BCa degenerate."""
    method = "BCa"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # promote BCa degeneracy warnings to errors
            res = stats.bootstrap(
                (diff,), np.median,
                n_resamples=n_resamples,
                method="BCa",
                random_state=seed,
                vectorized=False,
            )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("non-finite BCa CI")
    except Exception as e:
        method = "percentile"
        log.warning("%s: BCa CI degenerate (%s); falling back to percentile bootstrap.",
                    label, type(e).__name__)
        res = stats.bootstrap(
            (diff,), np.median,
            n_resamples=n_resamples,
            method="percentile",
            random_state=seed,
            vectorized=False,
        )
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    return lo, hi, method


def run_one(
    anatomy: str, R: int, metric: str, competitor: str,
    ours_vec: np.ndarray, comp_vec: np.ndarray,
    log: logging.Logger,
) -> dict | None:
    label = f"anatomy={anatomy} R={R} metric={metric} vs {competitor}"

    # Sanity-pair: drop NaNs in either vector (intersection step in load_data
    # should make this unnecessary, but be defensive).
    valid = np.isfinite(ours_vec) & np.isfinite(comp_vec)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        log.warning("%s: dropping %d slices with NaN in ours or competitor.",
                    label, n_dropped)
    ours_vec = ours_vec[valid]
    comp_vec = comp_vec[valid]

    n = ours_vec.size
    if n < 5:
        log.warning("%s: n=%d < 5, skipping.", label, n)
        return None

    sign = -1.0 if metric == "nrmse" else 1.0
    diff = sign * (ours_vec - comp_vec)  # positive => ours better

    median_diff = float(np.median(diff))
    mean_diff = float(np.mean(diff))
    if median_diff != 0.0 and mean_diff != 0.0 and np.sign(median_diff) != np.sign(mean_diff):
        log.info("%s: sign(median)=%+.0f != sign(mean)=%+.0f (median=%.6g, mean=%.6g)",
                 label, np.sign(median_diff), np.sign(mean_diff), median_diff, mean_diff)

    ci_lo, ci_hi, ci_method = median_ci(diff, log, label)

    sd = float(np.std(diff, ddof=1))
    if sd == 0.0 or not np.isfinite(sd):
        d = float("nan")
        d_lo = d_hi = float("nan")
    else:
        d = mean_diff / sd
        d_lo, d_hi = paired_d_ci(d, n)

    zeros = int((diff == 0).sum())
    if zeros > 0:
        log.info("%s: %d zero paired-differences (Wilcoxon will drop these).",
                 label, zeros)
    try:
        # SciPy default: zero_method='wilcox' (drop zeros).
        wres = stats.wilcoxon(diff, alternative="two-sided")
        wp = float(wres.pvalue)
    except ValueError as e:
        log.warning("%s: Wilcoxon failed (%s); p set NaN.", label, e)
        wp = float("nan")

    try:
        tres = stats.ttest_rel(ours_vec, comp_vec, alternative="two-sided")
        tp = float(tres.pvalue)
    except Exception as e:
        log.warning("%s: t-test failed (%s); p set NaN.", label, e)
        tp = float("nan")

    rb = rank_biserial(diff)

    direction = ("ours_better" if median_diff > 0
                 else "ours_worse" if median_diff < 0
                 else "tie")

    is_ssim = (metric == "ssim")
    return {
        "anatomy": anatomy,
        "acceleration": int(R),
        "metric": metric,
        "competitor": competitor,
        "n_slices": int(n),
        "mean_ours": float(np.mean(ours_vec)),
        "mean_competitor": float(np.mean(comp_vec)),
        "median_diff": median_diff,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "median_diff_pp": median_diff * 100.0 if is_ssim else float("nan"),
        "ci95_low_pp":    ci_lo       * 100.0 if is_ssim else float("nan"),
        "ci95_high_pp":   ci_hi       * 100.0 if is_ssim else float("nan"),
        "cohens_d": d,
        "d_ci_low": d_lo,
        "d_ci_high": d_hi,
        "rank_biserial": rb,
        "wilcoxon_p": wp,
        "ttest_p": tp,
        "direction": direction,
    }


def build_paired(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide so each row is one slice with one column per method."""
    wide = long_df.pivot_table(
        index=["anatomy", "acceleration", "metric", "slice_id"],
        columns="method",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def main() -> int:
    log = setup_logger()
    log.info("---- run_tests starting ----")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    end_to_end = cfg["methods"]["end_to_end"]            # claim 2 competitors
    other_diff = cfg["methods"]["other_diffusion"]       # claim 1 competitors

    if not LONG_PARQUET.exists():
        log.error("Missing %s; run load_data + select_ours first", LONG_PARQUET)
        return 1
    long_df = pd.read_parquet(LONG_PARQUET)
    wide = build_paired(long_df)

    if "ours" not in wide.columns:
        log.error("'ours' method missing in long_data; did select_ours run?")
        return 1

    metrics_all = ["ssim", "psnr", "nrmse"]
    accelerations = sorted(long_df["acceleration"].unique().tolist())
    anatomies = sorted(long_df["anatomy"].unique().tolist())

    claim1_rows: list[dict] = []
    claim2_rows: list[dict] = []

    for anatomy in anatomies:
        for R in accelerations:
            for metric in metrics_all:
                cell = wide[(wide["anatomy"] == anatomy)
                            & (wide["acceleration"] == R)
                            & (wide["metric"] == metric)]
                if cell.empty:
                    log.warning("No data for anatomy=%s R=%d metric=%s; skipping.",
                                anatomy, R, metric)
                    continue
                ours_vec = cell["ours"].to_numpy(dtype=float)

                # Claim 1: vs other diffusion (all 3 metrics).
                for comp in other_diff:
                    if comp not in cell.columns:
                        log.warning("competitor %s missing for anatomy=%s R=%d metric=%s",
                                    comp, anatomy, R, metric)
                        continue
                    comp_vec = cell[comp].to_numpy(dtype=float)
                    rec = run_one(anatomy, R, metric, comp, ours_vec, comp_vec, log)
                    if rec is not None:
                        claim1_rows.append(rec)

                # Claim 2: vs end-to-end, SSIM only.
                if metric == "ssim":
                    for comp in end_to_end:
                        if comp not in cell.columns:
                            log.warning("competitor %s missing for anatomy=%s R=%d metric=%s",
                                        comp, anatomy, R, metric)
                            continue
                        comp_vec = cell[comp].to_numpy(dtype=float)
                        rec = run_one(anatomy, R, metric, comp, ours_vec, comp_vec, log)
                        if rec is not None:
                            claim2_rows.append(rec)

    claim1_df = pd.DataFrame(claim1_rows, columns=OUTPUT_COLS).sort_values(
        ["anatomy", "metric", "competitor", "acceleration"]
    ).reset_index(drop=True)
    claim2_df = pd.DataFrame(claim2_rows, columns=OUTPUT_COLS).sort_values(
        ["anatomy", "competitor", "acceleration"]
    ).reset_index(drop=True)

    claim1_df.to_csv(CLAIM1_CSV, index=False)
    claim2_df.to_csv(CLAIM2_CSV, index=False)
    log.info("Wrote %s (%d rows) and %s (%d rows).",
             CLAIM1_CSV.name, len(claim1_df), CLAIM2_CSV.name, len(claim2_df))

    # Quick contradicting-cell summary in the log.
    contra1 = claim1_df[(claim1_df["direction"] == "ours_worse")
                        & (claim1_df["wilcoxon_p"] < cfg.get("alpha", 0.05))]
    contra2 = claim2_df[(claim2_df["direction"] == "ours_worse")
                        & (claim2_df["wilcoxon_p"] < cfg.get("alpha", 0.05))]
    log.info("Claim 1: %d/%d cells contradict claim (ours_worse AND p<alpha).",
             len(contra1), len(claim1_df))
    log.info("Claim 2: %d/%d cells contradict claim (ours_worse AND p<alpha).",
             len(contra2), len(claim2_df))

    log.info("---- run_tests done ----")
    return 0


if __name__ == "__main__":
    sys.exit(main())
