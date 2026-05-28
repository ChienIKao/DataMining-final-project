from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances


def compute_location_repeat_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute each user's average same-(weekday,t) modal-location ratio."""
    data = df.copy()
    data["weekday"] = data["d"] % 7
    group_counts = (
        data.groupby(["uid", "weekday", "t", "x", "y"]).size().rename("count").reset_index()
    )
    totals = group_counts.groupby(["uid", "weekday", "t"])["count"].sum().rename("total")
    max_counts = group_counts.groupby(["uid", "weekday", "t"])["count"].max().rename("max_count")
    ratios = pd.concat([max_counts, totals], axis=1).reset_index()
    ratios["ratio"] = ratios["max_count"] / ratios["total"]
    return ratios.groupby("uid")["ratio"].mean().rename("repeat_rate").reset_index()


def _entropy(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def compute_day_type_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekday/holiday spatial entropy and profile distance per user."""
    data = df.copy()
    if "is_holiday" not in data.columns:
        data["is_holiday"] = (data["d"] % 7).isin([5, 6])

    rows = []
    for uid, group in data.groupby("uid", sort=False):
        profiles = {}
        row = {"uid": int(uid)}
        for label, is_holiday in [("weekday", False), ("holiday", True)]:
            sub = group[group["is_holiday"].eq(is_holiday)]
            counts = sub.groupby(["x", "y"]).size()
            row[f"{label}_entropy"] = _entropy(counts)
            profiles[label] = counts
        all_cells = profiles["weekday"].index.union(profiles["holiday"].index)
        if len(all_cells) == 0:
            row["profile_diff"] = 0.0
        else:
            weekday_vec = profiles["weekday"].reindex(all_cells, fill_value=0).to_numpy().reshape(1, -1)
            holiday_vec = profiles["holiday"].reindex(all_cells, fill_value=0).to_numpy().reshape(1, -1)
            if weekday_vec.sum() == 0 or holiday_vec.sum() == 0:
                row["profile_diff"] = 1.0
            else:
                row["profile_diff"] = float(cosine_distances(weekday_vec, holiday_vec)[0, 0])
        rows.append(row)
    return pd.DataFrame(rows)


def classify_mobility_type(stability_df: pd.DataFrame, n_types: int = 4) -> pd.DataFrame:
    """Cluster users by stability features."""
    out = stability_df.copy()
    feature_cols = ["repeat_rate", "weekday_entropy", "holiday_entropy", "profile_diff"]
    if out.empty:
        out["mobility_type"] = pd.Series(dtype="int64")
        return out
    k = min(n_types, len(out))
    if k <= 1:
        out["mobility_type"] = 0
        return out
    features = out[feature_cols].fillna(0.0).to_numpy(dtype=float)
    out["mobility_type"] = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(features)
    return out


def build_user_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build and save user-level mobility stability features."""
    repeat = compute_location_repeat_rate(df)
    entropy = compute_day_type_entropy(df)
    stability = repeat.merge(entropy, on="uid", how="outer").fillna(0.0)
    stability = classify_mobility_type(stability)
    path = Path("data/user_stability_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    stability.to_csv(path, index=False)
    return stability
