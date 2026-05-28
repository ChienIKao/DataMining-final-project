from __future__ import annotations

import numpy as np
import pandas as pd

GRID_SIZE = 200
TRAIN_DAYS = 60  # d=1~60 training; d=61~75 test


def load_city(
    path: str,
    max_users: int | None = None,
    sample_users: int | None = None,
    random_seed: int = 42,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Load city trajectory CSV, filter out placeholder rows (x=999)."""
    dtype = {"uid": "int32", "d": "int16", "t": "int8", "x": "int16", "y": "int16"}

    if chunksize:
        chunks = pd.read_csv(path, dtype=dtype, chunksize=chunksize)
        df = pd.concat(
            [c[c["x"] != 999] for c in chunks],
            ignore_index=True,
        )
    else:
        df = pd.read_csv(path, dtype=dtype)
        df = df[df["x"] != 999].reset_index(drop=True)

    invalid_t = ~df["t"].between(0, 47)
    n_dropped = int(invalid_t.sum())
    if n_dropped > 0:
        print(f"[loader] dropped invalid t: {n_dropped} rows")
    df = df[~invalid_t].reset_index(drop=True)

    if max_users is not None:
        uids = sorted(df["uid"].unique())[:max_users]
        df = df[df["uid"].isin(uids)].reset_index(drop=True)
    elif sample_users is not None:
        all_uids = df["uid"].unique()
        rng = np.random.default_rng(random_seed)
        sampled = rng.choice(all_uids, size=min(sample_users, len(all_uids)), replace=False)
        df = df[df["uid"].isin(sampled)].reset_index(drop=True)

    print(f"[loader] loaded {len(df):,} rows, {df['uid'].nunique():,} users")
    return df


def split_train_test(
    df: pd.DataFrame,
    train_days: int = TRAIN_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by day: train d<=train_days, test d>train_days."""
    train = df[df["d"] <= train_days].reset_index(drop=True)
    test = df[df["d"] > train_days].reset_index(drop=True)
    return train, test


def split_train_val_test(
    df: pd.DataFrame,
    val_start: int = 51,
    train_days: int = TRAIN_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way split: train d<val_start, val val_start<=d<=train_days, test d>train_days."""
    train = df[df["d"] < val_start].reset_index(drop=True)
    val = df[(df["d"] >= val_start) & (df["d"] <= train_days)].reset_index(drop=True)
    test = df[df["d"] > train_days].reset_index(drop=True)
    return train, val, test
