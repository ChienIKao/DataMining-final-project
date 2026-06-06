from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

GRID_SIZE = 200
GEOBLEU_BETA = 0.5
GEOBLEU_N = 5
REQUIRED_COLUMNS = ["uid", "d", "t", "x", "y"]


def _require_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_submission(generated: pd.DataFrame, train_df: pd.DataFrame) -> bool:
    """Validate basic trajectory output constraints."""
    _require_columns(generated)
    train_uids = set(train_df["uid"].unique())
    unknown_uids = set(generated["uid"].unique()) - train_uids
    if unknown_uids:
        raise ValueError(f"Generated output contains unknown uid(s), sample={list(unknown_uids)[:5]}")
    if not generated["t"].between(0, 47).all():
        raise ValueError("Generated output contains t outside [0, 47]")
    if not generated["x"].between(0, GRID_SIZE - 1).all():
        raise ValueError("Generated output contains x outside grid")
    if not generated["y"].between(0, GRID_SIZE - 1).all():
        raise ValueError("Generated output contains y outside grid")
    return True


def compute_geobleu(
    generated: pd.DataFrame,
    reference: pd.DataFrame,
    processes: int = 4,
) -> dict:
    """Compute GEO-BLEU using geobleu library bulk method."""
    _require_columns(generated)
    _require_columns(reference)
    import geobleu

    generated = generated[REQUIRED_COLUMNS].sort_values(REQUIRED_COLUMNS)
    reference = reference[REQUIRED_COLUMNS].sort_values(REQUIRED_COLUMNS)

    n_users = generated["uid"].nunique()
    print(f"[metrics] converting {len(generated):,} rows to records …")
    t0 = time.time()
    gen_list = [tuple(r) for r in generated.itertuples(index=False)]
    ref_list = [tuple(r) for r in reference.itertuples(index=False)]
    print(f"[metrics] conversion done ({time.time()-t0:.1f}s), running bulk GEO-BLEU ({n_users:,} users, {processes} workers) …")
    t1 = time.time()
    score = geobleu.calc_geobleu_bulk(gen_list, ref_list, processes=processes)
    print(f"[metrics] GEO-BLEU done: mean={score:.5f}  ({time.time()-t1:.1f}s)")
    return {"mean": float(score), "per_user": {}}


def compute_fde(generated: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Compute final displacement error for each (uid, d)."""
    _require_columns(generated)
    _require_columns(reference)
    gen_last = generated.sort_values("t").groupby(["uid", "d"], as_index=False).tail(1)
    ref_last = reference.sort_values("t").groupby(["uid", "d"], as_index=False).tail(1)
    merged = gen_last.merge(ref_last, on=["uid", "d"], suffixes=("_gen", "_ref"))
    if merged.empty:
        return {"mean": 0.0, "per_user_day": {}}
    dist = np.sqrt((merged["x_gen"] - merged["x_ref"]) ** 2 + (merged["y_gen"] - merged["y_ref"]) ** 2)
    per_user_day = {
        f"{int(row.uid)}:{int(row.d)}": float(value)
        for row, value in zip(merged.itertuples(index=False), dist)
    }
    return {"mean": float(dist.mean()), "per_user_day": per_user_day}


def generate_report(
    method_name: str,
    geobleu_result: dict,
    fde_result: dict,
    save_path: str = "eval/reports/",
) -> None:
    """Save JSON and CSV reports for one model run."""
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method_name,
        "geobleu_mean": float(geobleu_result.get("mean", 0.0)),
        "fde_mean": float(fde_result.get("mean", 0.0)),
        "per_user_geobleu": geobleu_result.get("per_user", {}),
    }
    (path / f"{method_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"method": method_name, "geobleu_mean": payload["geobleu_mean"], "fde_mean": payload["fde_mean"]}]
    ).to_csv(path / f"{method_name}.csv", index=False)
