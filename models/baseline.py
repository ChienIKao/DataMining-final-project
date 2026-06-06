from __future__ import annotations

import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from eval.metrics import compute_fde, compute_geobleu, generate_report

GRID_SIZE = 200
TIME_STEPS = 48


def _prediction_index(train_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    uids = np.sort(train_df["uid"].unique())
    n_u, n_d, n_t = len(uids), len(test_days), TIME_STEPS
    uid_arr = np.repeat(uids, n_d * n_t)
    d_arr = np.tile(np.repeat(np.array(test_days, dtype="int16"), n_t), n_u)
    t_arr = np.tile(np.arange(n_t, dtype="int8"), n_u * n_d)
    return pd.DataFrame({"uid": uid_arr, "d": d_arr, "t": t_arr})


def _group_mode(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Return a DataFrame with columns keys + ['x', 'y'] giving the modal (x,y) per group."""
    return (
        data.groupby(keys + ["x", "y"], sort=False)
        .size()
        .reset_index(name="_cnt")
        .sort_values("_cnt", ascending=False)
        .drop_duplicates(subset=keys)[keys + ["x", "y"]]
        .reset_index(drop=True)
    )


def predict_per_user_mode(train_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    """Predict each user's most frequent historical (weekday, t) location (vectorised)."""
    print("[baseline] per_user_mode: building mode tables …")
    t0 = time.time()
    data = train_df.copy()
    data["weekday"] = (data["d"] % 7).astype("int8")

    t1 = _group_mode(data, ["uid", "weekday", "t"])
    t2 = _group_mode(data, ["uid", "t"]).rename(columns={"x": "x2", "y": "y2"})
    t3 = _group_mode(data, ["weekday", "t"]).rename(columns={"x": "x3", "y": "y3"})
    gx = int(data["x"].mode().iloc[0])
    gy = int(data["y"].mode().iloc[0])
    print(f"[baseline] mode tables built in {time.time()-t0:.1f}s — building prediction index …")

    pred = _prediction_index(train_df, test_days)
    pred["weekday"] = (pred["d"] % 7).astype("int8")

    print("[baseline] merging fallback levels …")
    pred = pred.merge(t1, on=["uid", "weekday", "t"], how="left")
    pred = pred.merge(t2, on=["uid", "t"], how="left")
    pred = pred.merge(t3, on=["weekday", "t"], how="left")

    pred["x"] = pred["x"].fillna(pred["x2"]).fillna(pred["x3"]).fillna(gx).astype("int16")
    pred["y"] = pred["y"].fillna(pred["y2"]).fillna(pred["y3"]).fillna(gy).astype("int16")
    pred = pred[["uid", "d", "t", "x", "y"]]
    print(f"[baseline] per_user_mode done in {time.time()-t0:.1f}s total")
    return pred.astype({"uid": "int64", "d": "int16", "t": "int8", "x": "int16", "y": "int16"})


def predict_per_user_mean(train_df: pd.DataFrame, test_days: list[int]) -> pd.DataFrame:
    """Predict rounded historical mean location (vectorised)."""
    print("[baseline] per_user_mean: building mean tables …")
    t0 = time.time()
    data = train_df.copy()
    data["weekday"] = (data["d"] % 7).astype("int8")

    def mean_table(keys: list[str], sx: str = "x", sy: str = "y") -> pd.DataFrame:
        return (
            data.groupby(keys)[["x", "y"]]
            .mean()
            .round()
            .astype(int)
            .reset_index()
            .rename(columns={"x": sx, "y": sy})
        )

    t1 = mean_table(["uid", "weekday", "t"])
    t2 = mean_table(["uid", "t"], "x2", "y2")
    t3 = mean_table(["weekday", "t"], "x3", "y3")
    gx = int(data["x"].mean().round())
    gy = int(data["y"].mean().round())

    pred = _prediction_index(train_df, test_days)
    pred["weekday"] = (pred["d"] % 7).astype("int8")

    pred = pred.merge(t1, on=["uid", "weekday", "t"], how="left")
    pred = pred.merge(t2, on=["uid", "t"], how="left")
    pred = pred.merge(t3, on=["weekday", "t"], how="left")

    pred["x"] = pred["x"].fillna(pred["x2"]).fillna(pred["x3"]).fillna(gx).clip(0, GRID_SIZE - 1).astype("int16")
    pred["y"] = pred["y"].fillna(pred["y2"]).fillna(pred["y3"]).fillna(gy).clip(0, GRID_SIZE - 1).astype("int16")
    pred = pred[["uid", "d", "t", "x", "y"]]
    print(f"[baseline] per_user_mean done in {time.time()-t0:.1f}s total")
    return pred.astype({"uid": "int64", "d": "int16", "t": "int8", "x": "int16", "y": "int16"})


def predict_bigram(train_df: pd.DataFrame, test_days: list[int], top_p: float = 1.0) -> pd.DataFrame:
    """Predict trajectories from per-user transition counts."""
    mode_pred = predict_per_user_mode(train_df, test_days)
    rng = np.random.default_rng(42)
    transitions: dict[tuple[int, int, int, int], Counter] = defaultdict(Counter)
    print("[baseline] bigram: building transition table …")
    uids_sorted = train_df["uid"].unique()
    for uid, group in tqdm(train_df.sort_values(["uid", "d", "t"]).groupby("uid", sort=False),
                           total=len(uids_sorted), desc="bigram build", unit="user"):
        coords = group[["x", "y", "t"]].to_numpy()
        for prev, nxt in zip(coords[:-1], coords[1:]):
            transitions[(int(uid), int(prev[0]), int(prev[1]), int(nxt[2]))][(int(nxt[0]), int(nxt[1]))] += 1

    pred = mode_pred.copy()
    label = f"bigram top_p={top_p}" if top_p < 1.0 else "bigram"
    for uid, user_idx in tqdm(pred.groupby("uid", sort=False).groups.items(),
                              total=pred["uid"].nunique(), desc=label, unit="user"):
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
    """Run baselines, evaluate, and write reports. Reuses saved prediction CSVs if present."""
    import pathlib
    pathlib.Path("eval/reports").mkdir(parents=True, exist_ok=True)
    test_days = sorted(test_df["d"].unique().astype(int).tolist())
    t_total = time.time()
    print(f"[baseline] running on {train_df['uid'].nunique():,} users, test days={test_days[0]}~{test_days[-1]}")

    model_fns = {
        "per_user_mode": lambda: predict_per_user_mode(train_df, test_days),
        "per_user_mean": lambda: predict_per_user_mean(train_df, test_days),
        "bigram": lambda: predict_bigram(train_df, test_days),
        "bigram_top_p07": lambda: predict_bigram(train_df, test_days, top_p=0.7),
    }
    scores = {}
    for name, fn in model_fns.items():
        t_eval = time.time()
        csv_path = f"eval/reports/{name}_predictions.csv"
        json_path = f"eval/reports/{name}.json"

        if pathlib.Path(json_path).exists():
            import json
            existing = json.loads(pathlib.Path(json_path).read_text())
            scores[name] = float(existing.get("geobleu_mean", 0.0))
            print(f"[baseline] {name}: skipped (cached GEO-BLEU={scores[name]:.5f})")
            continue

        if pathlib.Path(csv_path).exists():
            print(f"[baseline] {name}: loading predictions from {csv_path} …")
            pred = pd.read_csv(csv_path, dtype={"uid":"int64","d":"int16","t":"int8","x":"int16","y":"int16"})
        else:
            pred = fn()
            pred = align_prediction_to_reference(pred, test_df)
            pred.to_csv(csv_path, index=False)
            print(f"[baseline] {name}: predictions saved to {csv_path}")

        print(f"[baseline] evaluating {name} …")
        geobleu_result = compute_geobleu(pred, test_df)
        fde_result = compute_fde(pred, test_df)
        generate_report(name, geobleu_result, fde_result)
        scores[name] = float(geobleu_result.get("mean", 0.0))
        print(f"[baseline] {name}: GEO-BLEU={scores[name]:.5f}  ({time.time()-t_eval:.1f}s)")

    print(f"[baseline] all done in {time.time()-t_total:.1f}s total")
    return scores
