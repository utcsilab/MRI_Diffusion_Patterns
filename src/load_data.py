"""Load the 8 result CSVs into a single long-format parquet.

Output: results/long_data.parquet with columns
    (anatomy, method, acceleration, slice_id, metric, value)

Plus diagnostics appended to results/diagnostics.log.

DOGG rows are kept as separate methods labelled `ours_K{K}_BS{BS}`; the
canonical "ours" row per (anatomy, acceleration) is chosen later by
src/select_ours.py.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LONG_PARQUET = RESULTS / "long_data.parquet"
DIAG_LOG = RESULTS / "diagnostics.log"

EXPECTED_R = {4, 8, 12, 16, 20}
METRIC_TOKENS = ("ssim", "psnr", "nrmse")
META_COLS = {
    "name", "path", "pattern", "anatomy", "acceleration", "recon",
    "mean ssim", "std ssim", "mean psnr", "std psnr", "mean nrmse", "std nrmse",
}

# anatomy key in config -> expected token in the CSV `anatomy` column (case-insensitive).
ANATOMY_TOKEN = {
    "t2_brain":  "brain",
    "pd_knee":   "pd",
    "pdfs_knee": "pdfs",
    "all_knee":  "combined",
}

DOGG_NAME_RE = re.compile(r"_K(\d+)_BS(\d+)$", re.IGNORECASE)
PER_SLICE_SUFFIX_RE = re.compile(r"^(.*)\s+(ssim|psnr|nrmse)$", re.IGNORECASE)


def setup_logger() -> logging.Logger:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("load_data")
    logger.setLevel(logging.INFO)
    # Reset handlers so re-runs in the same process don't double-log.
    logger.handlers.clear()
    fh = logging.FileHandler(DIAG_LOG, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


def derive_method(recon: str, pattern: str, name: str, log: logging.Logger) -> str | None:
    r = (recon or "").strip().lower()
    p = (pattern or "").strip()
    if r == "modl" and p == "LOUPE":
        return "modl+loupe"
    if r == "modl" and p == "Poisson":
        return "modl+poisson"
    if r == "diffusion" and p == "LOUPE":
        return "diffusion+loupe"
    if r == "diffusion" and p == "Poisson":
        return "diffusion+poisson"
    if r == "diffusion" and p == "DOGG":
        m = DOGG_NAME_RE.search(name or "")
        if not m:
            log.warning("DOGG row with unparseable K/BS in name=%r; skipping", name)
            return None
        return f"ours_K{int(m.group(1))}_BS{int(m.group(2))}"
    log.warning("Unknown (recon, pattern) combo: recon=%r pattern=%r name=%r; skipping",
                recon, pattern, name)
    return None


def split_per_slice_columns(columns: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Return ({slice_id: {metric: column_name}}, [skipped_columns])."""
    by_slice: dict[str, dict[str, str]] = {}
    skipped: list[str] = []
    for col in columns:
        if col in META_COLS:
            continue
        m = PER_SLICE_SUFFIX_RE.match(col)
        if not m:
            skipped.append(col)
            continue
        slice_id, metric = m.group(1).strip(), m.group(2).lower()
        by_slice.setdefault(slice_id, {})[metric] = col
    return by_slice, skipped


def load_csv(path: Path, log: logging.Logger) -> pd.DataFrame:
    # The CSVs have an unnamed leading index column.
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


def process_anatomy(
    anatomy_key: str,
    diffusion_path: Path,
    modl_path: Path,
    log: logging.Logger,
) -> pd.DataFrame:
    log.info("=== anatomy=%s ===", anatomy_key)
    diff_df = load_csv(diffusion_path, log)
    modl_df = load_csv(modl_path, log)

    # Sanity-check anatomy column.
    expected_tok = ANATOMY_TOKEN[anatomy_key]
    for src_path, df in [(diffusion_path, diff_df), (modl_path, modl_df)]:
        toks = {str(x).strip().lower() for x in df["anatomy"].unique()}
        if toks != {expected_tok}:
            log.warning("Anatomy mismatch in %s: expected {%s}, got %s",
                        src_path.name, expected_tok, toks)

    # Build per-slice column maps for each file.
    diff_by_slice, diff_skipped = split_per_slice_columns(list(diff_df.columns))
    modl_by_slice, modl_skipped = split_per_slice_columns(list(modl_df.columns))
    if diff_skipped:
        log.warning("%s: %d non-conforming column names skipped (sample: %s)",
                    diffusion_path.name, len(diff_skipped), diff_skipped[:3])
    if modl_skipped:
        log.warning("%s: %d non-conforming column names skipped (sample: %s)",
                    modl_path.name, len(modl_skipped), modl_skipped[:3])

    # Index rows by (R, method).
    rows: list[tuple[int, str, pd.Series, dict[str, dict[str, str]]]] = []
    for df, by_slice, src in [(diff_df, diff_by_slice, "diffusion"),
                              (modl_df, modl_by_slice, "modl")]:
        for _, row in df.iterrows():
            method = derive_method(row.get("recon", ""), row.get("pattern", ""),
                                   row.get("name", ""), log)
            if method is None:
                continue
            try:
                r = int(row["acceleration"])
            except (TypeError, ValueError):
                log.warning("%s: row name=%r has non-integer acceleration=%r; skipping",
                            src, row.get("name"), row.get("acceleration"))
                continue
            rows.append((r, method, row, by_slice))

    # Group by acceleration, do slice-intersection check, then melt.
    long_records: list[dict] = []
    by_r: dict[int, list[tuple[str, pd.Series, dict[str, dict[str, str]]]]] = {}
    for r, m, row, by_slice in rows:
        by_r.setdefault(r, []).append((m, row, by_slice))

    missing_r = EXPECTED_R - set(by_r.keys())
    if missing_r:
        log.warning("anatomy=%s: missing accelerations %s entirely", anatomy_key, sorted(missing_r))

    for r in sorted(by_r.keys()):
        method_groups = by_r[r]
        slice_sets = [set(bs.keys()) for _, _, bs in method_groups]
        union = set().union(*slice_sets)
        intersection = set.intersection(*slice_sets) if slice_sets else set()
        if any(s != union for s in slice_sets):
            dropped = union - intersection
            log.warning("anatomy=%s R=%d: slice-set mismatch; using intersection of "
                        "%d slices (dropped %d). Methods present: %s",
                        anatomy_key, r, len(intersection), len(dropped),
                        sorted({m for m, _, _ in method_groups}))
            log.warning("  dropped slice IDs (up to 10 shown): %s",
                        sorted(dropped)[:10])
        else:
            log.info("anatomy=%s R=%d: %d methods, %d slices matched",
                     anatomy_key, r, len(method_groups), len(intersection))

        # Verify all 5 non-DOGG methods present (DOGG can be multiple).
        present_methods = {m for m, _, _ in method_groups}
        non_dogg = {m for m in present_methods if not m.startswith("ours_K")}
        expected_non_dogg = {"modl+loupe", "modl+poisson",
                             "diffusion+loupe", "diffusion+poisson"}
        missing = expected_non_dogg - non_dogg
        if missing:
            log.warning("anatomy=%s R=%d: missing methods %s",
                        anatomy_key, r, sorted(missing))
        if not any(m.startswith("ours_K") for m in present_methods):
            log.warning("anatomy=%s R=%d: no DOGG candidates found", anatomy_key, r)

        # Emit long rows over the intersection.
        for method, row, by_slice in method_groups:
            for slice_id in intersection:
                metric_cols = by_slice[slice_id]
                for metric in METRIC_TOKENS:
                    col = metric_cols.get(metric)
                    if col is None:
                        log.warning("anatomy=%s R=%d method=%s slice=%s: metric %s missing",
                                    anatomy_key, r, method, slice_id, metric)
                        continue
                    val = row[col]
                    long_records.append({
                        "anatomy": anatomy_key,
                        "method": method,
                        "acceleration": r,
                        "slice_id": slice_id,
                        "metric": metric,
                        "value": float(val) if pd.notna(val) else float("nan"),
                    })

    return pd.DataFrame.from_records(long_records)


def main() -> int:
    log = setup_logger()
    log.info("---- load_data starting ----")

    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    data_files = cfg["data_files"]

    frames = []
    for anatomy_key, paths in data_files.items():
        diff_path = ROOT / paths["diffusion"]
        modl_path = ROOT / paths["modl"]
        for p in (diff_path, modl_path):
            if not p.exists():
                log.error("Missing CSV: %s", p)
                return 1
        frames.append(process_anatomy(anatomy_key, diff_path, modl_path, log))

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.astype({
        "anatomy": "string",
        "method": "string",
        "acceleration": "int32",
        "slice_id": "string",
        "metric": "string",
        "value": "float64",
    })
    RESULTS.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(LONG_PARQUET, index=False)
    log.info("Wrote %s with %d rows", LONG_PARQUET, len(long_df))
    log.info("---- load_data done ----")
    return 0


if __name__ == "__main__":
    sys.exit(main())
