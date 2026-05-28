from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from eval.metrics import compute_fde, compute_geobleu, generate_report

GRID_SIZE = 200
TIME_STEPS = 48


def _mode_xy(group: pd.DataFrame) -> tuple[int, int]:
    counts = group.groupby(["x", "y"]).size()
    x, y = counts.sort_values(ascending=False).index[0]
    return int(x), int(y)


def _prediction_index(train_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    uids = np.sort(train_df["uid"].unique())
    rows = [(int(uid), int(d), int(t)) for uid in uids for d in test_days for t in range(TIME_STEPS)]
    return pd.DataFrame(rows, columns=["uid", "d", "t"])


def _fallback_tables(train_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    data = train_df.copy()
    data["weekday"] = data["d"] % 7
    by_user_weekday_t = {
        key: _mode_xy(group)
        for key, group in data.groupby(["uid", "weekday", "t"], sort=False)
    }
    by_user_t = {
        key: _mode_xy(group)
        for key, group in data.groupby(["uid", "t"], sort=False)
    }
    by_weekday_t = {
        key: _mode_xy(group)
        for key, group in data.groupby(["weekday", "t"], sort=False)
    }
    global_xy = _mode_xy(data)
    by_weekday_t[("__global__", "__global__")] = global_xy
    return by_user_weekday_t, by_user_t, by_weekday_t


def predict_per_user_mode(train_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    """Predict each user's most frequent historical (weekday, t) location."""
    by_user_weekday_t, by_user_t, by_weekday_t = _fallback_tables(train_df)
    global_xy = by_weekday_t[("__global__", "__global__")]
    pred = _prediction_index(train_df, test_days)
    xs, ys = [], []
    for uid, d, t in pred[["uid", "d", "t"]].itertuples(index=False):
        weekday = d % 7
        xy = (
            by_user_weekday_t.get((uid, weekday, t))
            or by_user_t.get((uid, t))
            or by_weekday_t.get((weekday, t))
            or global_xy
        )
        xs.append(xy[0])
        ys.append(xy[1])
    pred["x"] = xs
    pred["y"] = ys
    return pred.astype({"uid": "int64", "d": "int16", "t": "int8", "x": "int16", "y": "int16"})


def predict_per_user_mean(train_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    """Predict rounded historical mean location with the same fallback levels as mode."""
    data = train_df.copy()
    data["weekday"] = data["d"] % 7
    tables = [
        data.groupby(["uid", "weekday", "t"])[["x", "y"]].mean().round().astype(int),
        data.groupby(["uid", "t"])[["x", "y"]].mean().round().astype(int),
        data.groupby(["weekday", "t"])[["x", "y"]].mean().round().astype(int),
    ]
    global_xy = tuple(data[["x", "y"]].mean().round().astype(int))
    pred = _prediction_index(train_df, test_days)
    xs, ys = [], []
    for uid, d, t in pred[["uid", "d", "t"]].itertuples(index=False):
        weekday = d % 7
        xy = None
        for table, key in zip(tables, [(uid, weekday, t), (uid, t), (weekday, t)]):
            if key in table.index:
                row = table.loc[key]
                xy = (int(row["x"]), int(row["y"]))
                break
        if xy is None:
            xy = global_xy
        xs.append(int(np.clip(xy[0], 0, GRID_SIZE - 1)))
        ys.append(int(np.clip(xy[1], 0, GRID_SIZE - 1)))
    pred["x"] = xs
    pred["y"] = ys
    return pred


def predict_bigram(train_df: pd.DataFrame, test_days: list[int], top_p: float = 1.0) -> pd.DataFrame:
    """Predict trajectories from per-user transition counts."""
    mode_pred = predict_per_user_mode(train_df, test_days)
    rng = np.random.default_rng(42)
    transitions: dict[tuple[int, int, int, int], Counter] = defaultdict(Counter)
    for uid, group in train_df.sort_values(["uid", "d", "t"]).groupby("uid", sort=False):
        coords = group[["x", "y", "t"]].to_numpy()
        for prev, nxt in zip(coords[:-1], coords[1:]):
            transitions[(int(uid), int(prev[0]), int(prev[1]), int(nxt[2]))][(int(nxt[0]), int(nxt[1]))] += 1

    pred = mode_pred.copy()
    for uid, user_idx in pred.groupby("uid", sort=False).groups.items():
        rows = pred.loc[user_idx].sort_values(["d", "t"])
        prev_xy: tuple[int, int] | None = None
        for idx, row in rows.iterrows():
            if row["t"] == 0 or prev_xy is None:
                prev_xy = (int(row["x"]), int(row["y"]))
                continue
            candidates = transitions.get((int(uid), prev_xy[0], prev_xy[1], int(row["t"])))
            if candidates:
                items = candidates.most_common()
                if top_p < 1.0:
                    counts = np.array([count for _, count in items], dtype=float)
                    probs = counts / counts.sum()
                    cutoff = np.searchsorted(np.cumsum(probs), top_p, side="right") + 1
                    items = items[:cutoff]
                    probs = np.array([count for _, count in items], dtype=float)
                    probs /= probs.sum()
                    choice = items[int(rng.choice(len(items), p=probs))][0]
                else:
                    choice = items[0][0]
                pred.at[idx, "x"] = choice[0]
                pred.at[idx, "y"] = choice[1]
                prev_xy = choice
            else:
                prev_xy = (int(row["x"]), int(row["y"]))
    return pred


def align_prediction_to_reference(pred: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Keep predictions at the same (uid, d, t) keys as the reference trajectory."""
    keys = reference[["uid", "d", "t"]].drop_duplicates()
    aligned = keys.merge(pred, on=["uid", "d", "t"], how="left")
    missing = aligned["x"].isna() | aligned["y"].isna()
    if missing.any():
        fallback = pred.groupby("uid")[["x", "y"]].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 0)
        for idx, row in aligned.loc[missing, ["uid"]].iterrows():
            if row["uid"] in fallback.index:
                aligned.at[idx, "x"] = fallback.loc[row["uid"], "x"]
                aligned.at[idx, "y"] = fallback.loc[row["uid"], "y"]
            else:
                aligned.at[idx, "x"] = 0
                aligned.at[idx, "y"] = 0
    return aligned[["uid", "d", "t", "x", "y"]].astype(
        {"uid": "int64", "d": "int16", "t": "int8", "x": "int16", "y": "int16"}
    )


def run_all_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    """Run baselines, evaluate, and write reports."""
    test_days = sorted(test_df["d"].unique().astype(int).tolist())
    predictions = {
        "per_user_mode": predict_per_user_mode(train_df, test_days),
        "per_user_mean": predict_per_user_mean(train_df, test_days),
        "bigram": predict_bigram(train_df, test_days),
        "bigram_top_p07": predict_bigram(train_df, test_days, top_p=0.7),
    }
    scores = {}
    for name, pred in predictions.items():
        pred = align_prediction_to_reference(pred, test_df)
        pred.to_csv(f"eval/reports/{name}_predictions.csv", index=False)
        geobleu_result = compute_geobleu(pred, test_df)
        fde_result = compute_fde(pred, test_df)
        generate_report(name, geobleu_result, fde_result)
        scores[name] = float(geobleu_result.get("mean", 0.0))
    return scores
