"""Generate CVAE result visualizations for the report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP_ACTUAL = "#2196F3"   # 藍
CMAP_CVAE   = "#F44336"   # 紅

# Per-User Mode 已知評估指標（100K 使用者）
MODE_FDE_GRIDS = 8.86
MODE_FDE_KM = MODE_FDE_GRIDS * 0.5


def load_data(city_path: str):
    dtype = {"uid": "int32", "d": "int16", "t": "int8", "x": "int16", "y": "int16"}
    grid = pd.read_csv("data/grid_to_latlon.csv")[["x", "y", "lat", "lon"]]
    grid_map = {(int(r.x), int(r.y)): (r.lat, r.lon) for r in grid.itertuples(index=False)}

    cvae = pd.read_csv("eval/reports/cvae_predictions.csv", dtype=dtype)

    print(f"Loading test data from {city_path} ...")
    actual = pd.read_csv(city_path, dtype=dtype)
    actual = actual[(actual["x"] != 999) & (actual["d"] > 60)]
    cvae_uids = set(cvae["uid"].unique())
    actual = actual[actual["uid"].isin(cvae_uids)].reset_index(drop=True)

    return cvae, actual, grid_map


def to_latlon(df: pd.DataFrame, grid_map: dict) -> pd.DataFrame:
    df = df.copy()
    lats, lons = [], []
    for _, row in df.iterrows():
        ll = grid_map.get((int(row["x"]), int(row["y"])), (None, None))
        lats.append(ll[0])
        lons.append(ll[1])
    df["lat"] = lats
    df["lon"] = lons
    return df.dropna(subset=["lat", "lon"])


# ── Figure 1: 3 使用者的單日軌跡比較（CVAE 預測 vs 實際）─────────────────────────

def fig_trajectory_comparison(cvae, actual, grid_map, n_users=3, test_day=61):
    uids = sorted(set(cvae["uid"]) & set(actual["uid"]))
    day_actual = actual[actual["d"] == test_day]
    spreads = (
        day_actual.groupby("uid")[["x", "y"]]
        .std().fillna(0).sum(axis=1)
        .reindex(uids).dropna()
        .sort_values()
    )
    if len(spreads) < n_users:
        selected = spreads.index[:n_users].tolist()
    else:
        idxs = np.linspace(0, len(spreads) - 1, n_users, dtype=int)
        selected = spreads.iloc[idxs].index.tolist()

    fig, axes = plt.subplots(1, n_users, figsize=(5 * n_users, 5))
    if n_users == 1:
        axes = [axes]

    labels = ["穩定型", "中等活動", "活躍型"]
    for ax, uid, label in zip(axes, selected, labels):
        def get_day_traj(df, u, d):
            sub = df[(df["uid"] == u) & (df["d"] == d)].sort_values("t")
            return to_latlon(sub, grid_map)

        act = get_day_traj(actual, uid, test_day)
        pred_c = get_day_traj(cvae, uid, test_day)

        for traj, color, lw, zorder in [
            (act,    CMAP_ACTUAL, 2.0, 3),
            (pred_c, CMAP_CVAE,   1.5, 2),
        ]:
            if len(traj) < 2:
                continue
            ax.plot(traj["lon"], traj["lat"], color=color, linewidth=lw,
                    alpha=0.85, zorder=zorder)
            ax.scatter(traj["lon"].iloc[0],  traj["lat"].iloc[0],
                       color=color, s=60, marker="o", zorder=zorder+1)
            ax.scatter(traj["lon"].iloc[-1], traj["lat"].iloc[-1],
                       color=color, s=80, marker="*", zorder=zorder+1)

        ax.set_title(f"uid={uid}  ({label})", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(labelsize=8)

    patches = [
        mpatches.Patch(color=CMAP_ACTUAL, label="Actual Trajectory"),
        mpatches.Patch(color=CMAP_CVAE,   label="CVAE Prediction"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"CVAE Predicted vs Actual Trajectories (Test Day d={test_day})", fontsize=13, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "cvae_trajectory_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {out}")


# ── Figure 2: 預測熱力圖 vs 實際熱力圖（所有使用者聚合）────────────────────────

def fig_density_heatmap(cvae, actual):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["CVAE Predicted Distribution", "Actual Distribution (Test Set)"]
    dfs    = [cvae, actual[actual["d"] > 60]]

    for ax, df, title in zip(axes, dfs, titles):
        counts, xedges, yedges = np.histogram2d(
            df["x"], df["y"],
            bins=50,
            range=[[0, 200], [0, 200]],
        )
        im = ax.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            cmap="hot",
            interpolation="bilinear",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        plt.colorbar(im, ax=ax, label="Visit Count")

    fig.suptitle("Spatial Distribution Density: CVAE vs Actual", fontsize=13)
    plt.tight_layout()
    out = OUTPUT_DIR / "cvae_density_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {out}")


# ── Figure 3: 預測終點誤差（FDE）分布─────────────────────────────────────────

def fig_fde_distribution(cvae, actual):
    def compute_fde_km(pred, ref):
        p_last = pred.sort_values("t").groupby(["uid", "d"]).tail(1)
        r_last = ref.sort_values("t").groupby(["uid", "d"]).tail(1)
        merged = p_last.merge(r_last, on=["uid", "d"], suffixes=("_p", "_r"))
        dist = np.sqrt((merged["x_p"] - merged["x_r"])**2 +
                       (merged["y_p"] - merged["y_r"])**2) * 0.5  # grids -> km
        return dist.dropna()

    fde_cvae = compute_fde_km(cvae, actual[actual["d"] > 60])
    cvae_mean_km = fde_cvae.mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    upper = fde_cvae.quantile(0.95)
    bins = np.linspace(0, upper, 40)
    ax.hist(fde_cvae, bins=bins, alpha=0.75, color=CMAP_CVAE,
            label=f"CVAE  mean={cvae_mean_km:.2f} km")
    ax.axvline(MODE_FDE_KM, color="#4CAF50", linewidth=2, linestyle="--",
               label=f"Per-User Mode mean={MODE_FDE_KM:.2f} km")
    ax.set_xlabel("Final Displacement Error (km)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("FDE Distribution: CVAE vs Per-User Mode Baseline", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = OUTPUT_DIR / "cvae_fde_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-path", default="raw_data/nagoya_challengedata.csv")
    parser.add_argument("--test-day", type=int, default=61)
    args = parser.parse_args()

    print("Loading data ...")
    cvae, actual, grid_map = load_data(args.city_path)

    print("Figure 1: trajectory comparison ...")
    fig_trajectory_comparison(cvae, actual, grid_map, test_day=args.test_day)

    print("Figure 2: density heatmap ...")
    fig_density_heatmap(cvae, actual)

    print("Figure 3: FDE distribution ...")
    fig_fde_distribution(cvae, actual)

    print("Done! All figures saved to docs/figures/")
