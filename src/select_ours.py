"""Select the canonical "ours" DOGG row per (anatomy, acceleration).

For each (anatomy, R), the manuscript's rule is: among the DOGG rows in the
diffusion CSV, pick the one with the smallest `mean nrmse`. Halt if that
column is missing or NaN.

Outputs:
  - results/ours_selection.csv  (auditable: winner K/BS + losers)
  - rewrites results/long_data.parquet so DOGG rows for the winner are
    relabelled `ours` and the other DOGG candidates are dropped.

Diagnostics are appended to results/diagnostics.log.
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
SELECTION_CSV = RESULTS / "ours_selection.csv"
DIAG_LOG = RESULTS / "diagnostics.log"

DOGG_NAME_RE = re.compile(r"_K(\d+)_BS(\d+)$", re.IGNORECASE)


def setup_logger() -> logging.Logger:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("select_ours")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(DIAG_LOG, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


def parse_kbs(name: str) -> tuple[int, int]:
    m = DOGG_NAME_RE.search(name or "")
    if not m:
        raise ValueError(f"Could not parse K/BS from name {name!r}")
    return int(m.group(1)), int(m.group(2))


def load_diffusion_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


def select_for_anatomy(anatomy_key: str, diff_path: Path, log: logging.Logger) -> list[dict]:
    df = load_diffusion_csv(diff_path)
    dogg = df[df["pattern"].astype(str).str.upper() == "DOGG"].copy()
    if dogg.empty:
        log.error("anatomy=%s: no DOGG rows in %s", anatomy_key, diff_path.name)
        raise RuntimeError(f"no DOGG rows for anatomy={anatomy_key}")

    # Verify mean nrmse is present and finite for every DOGG row.
    if "mean nrmse" not in dogg.columns:
        raise RuntimeError(f"anatomy={anatomy_key}: 'mean nrmse' column missing in {diff_path}")
    nan_rows = dogg[dogg["mean nrmse"].isna()]
    if not nan_rows.empty:
        names = nan_rows["name"].tolist()
        raise RuntimeError(
            f"anatomy={anatomy_key}: NaN 'mean nrmse' in DOGG rows: {names}"
        )

    # Parse K/BS for every DOGG row.
    dogg["K"] = dogg["name"].map(lambda n: parse_kbs(n)[0])
    dogg["BS"] = dogg["name"].map(lambda n: parse_kbs(n)[1])
    dogg["acceleration"] = dogg["acceleration"].astype(int)

    rows: list[dict] = []
    for r, group in dogg.groupby("acceleration"):
        winner_idx = group["mean nrmse"].idxmin()
        winner = group.loc[winner_idx]
        losers = group.drop(index=winner_idx)
        loser_strs = [
            f"K{int(rr['K'])}_BS{int(rr['BS'])}={float(rr['mean nrmse']):.6f}"
            for _, rr in losers.iterrows()
        ]
        rec = {
            "anatomy": anatomy_key,
            "acceleration": int(r),
            "chosen_K": int(winner["K"]),
            "chosen_BS": int(winner["BS"]),
            "chosen_name": str(winner["name"]),
            "chosen_mean_nrmse": float(winner["mean nrmse"]),
            "n_candidates": int(len(group)),
            "losers": "; ".join(loser_strs) if loser_strs else "",
        }
        rows.append(rec)

        log.info(
            "anatomy=%s R=%d: chose K=%d BS=%d (mean nrmse=%.6f) over %d alternatives: [%s]",
            anatomy_key, r, rec["chosen_K"], rec["chosen_BS"],
            rec["chosen_mean_nrmse"], len(losers), rec["losers"],
        )

    return rows


def update_long_parquet(selection: pd.DataFrame, log: logging.Logger) -> None:
    long_df = pd.read_parquet(LONG_PARQUET)

    # Build the set of winner method labels per (anatomy, acceleration).
    winners: dict[tuple[str, int], str] = {}
    for _, row in selection.iterrows():
        key = (row["anatomy"], int(row["acceleration"]))
        winners[key] = f"ours_K{int(row['chosen_K'])}_BS{int(row['chosen_BS'])}"

    is_dogg = long_df["method"].str.startswith("ours_K", na=False)
    non_dogg = long_df[~is_dogg].copy()
    dogg_rows = long_df[is_dogg].copy()

    # Keep only winner DOGG rows; rename their method to "ours".
    keep_mask = pd.Series(False, index=dogg_rows.index)
    for (anatomy, r), winner_method in winners.items():
        sel = (
            (dogg_rows["anatomy"] == anatomy)
            & (dogg_rows["acceleration"] == r)
            & (dogg_rows["method"] == winner_method)
        )
        keep_mask = keep_mask | sel
    kept_dogg = dogg_rows[keep_mask].copy()
    kept_dogg["method"] = "ours"

    dropped_n = int((~keep_mask).sum())
    new_long = pd.concat([non_dogg, kept_dogg], ignore_index=True)

    # Cast to match load_data.py output schema.
    new_long = new_long.astype({
        "anatomy": "string",
        "method": "string",
        "acceleration": "int32",
        "slice_id": "string",
        "metric": "string",
        "value": "float64",
    })

    new_long.to_parquet(LONG_PARQUET, index=False)
    log.info(
        "Rewrote %s: dropped %d loser DOGG rows, relabelled %d winner rows to 'ours'. "
        "New total: %d rows.",
        LONG_PARQUET.name, dropped_n, len(kept_dogg), len(new_long),
    )


def main() -> int:
    log = setup_logger()
    log.info("---- select_ours starting ----")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    if not LONG_PARQUET.exists():
        log.error("Missing %s; run src/load_data.py first", LONG_PARQUET)
        return 1

    all_rows: list[dict] = []
    for anatomy_key, paths in cfg["data_files"].items():
        diff_path = ROOT / paths["diffusion"]
        all_rows.extend(select_for_anatomy(anatomy_key, diff_path, log))

    selection = pd.DataFrame(all_rows).sort_values(["anatomy", "acceleration"]).reset_index(drop=True)

    # Update the parquet first, then write the selection CSV last. The Makefile
    # treats SELECTION_CSV as this script's output stamp; if the parquet ended
    # up with a newer mtime than the CSV, `make` would re-run select_ours every
    # invocation.
    update_long_parquet(selection, log)

    selection.to_csv(SELECTION_CSV, index=False)
    log.info("Wrote %s with %d rows", SELECTION_CSV, len(selection))

    log.info("---- select_ours done ----")
    return 0


if __name__ == "__main__":
    sys.exit(main())
