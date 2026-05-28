"""
Run full analysis pipeline on Nagoya data and output statistics + figures.
Results are saved to output/figures/ and output/stats.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from data.loader import load_city, split_train_test
from data.preprocessing import label_holidays, BBOX, _LAT_STEP, _LON_STEP, GRID_SIZE
from analysis.eda import (
    plot_spatial_heatmap,
    plot_unique_users_map,
    plot_daily_active_users,
    plot_trajectory_map,
)
from analysis.clustering import compute_grid_density, run_hdbscan, assign_user_hotspots

# Nagoya Station real coordinates
NAGOYA_STATION = {"lat": 35.1709, "lon": 136.8815}
NAGOYA_STATION_GRID = {
    "x": round((NAGOYA_STATION["lat"] - BBOX["south"]) / _LAT_STEP),
    "y": round((NAGOYA_STATION["lon"] - BBOX["west"]) / _LON_STEP),
}


def _log(msg: str) -> None:
    print(f"[analysis] {msg}", flush=True)


# ── 1. Load ────────────────────────────────────────────────────────────────────
def step1_load(city_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _log("Loading data …")
    df = load_city(city_path)
    df = label_holidays(df)
    train_df, test_df = split_train_test(df)
    _log(f"Total rows={len(df):,}  users={df['uid'].nunique():,}  days={df['d'].nunique()}")
    _log(f"Train rows={len(train_df):,}  Test rows={len(test_df):,}")
    return df, train_df, test_df


# ── 2. Basic statistics ────────────────────────────────────────────────────────
def step2_basic_stats(df: pd.DataFrame) -> dict:
    _log("Computing basic statistics …")
    stats: dict = {}
    stats["total_rows"] = int(len(df))
    stats["total_users"] = int(df["uid"].nunique())
    stats["total_days"] = int(df["d"].nunique())

    daily = df.groupby("d")["uid"].nunique()
    stats["daily_active_mean"] = float(daily.mean())
    stats["daily_active_min"] = int(daily.min())
    stats["daily_active_max"] = int(daily.max())
    stats["daily_active_std"] = float(daily.std())

    # Holiday classification result
    holiday_days = df[df["is_holiday"] == 1]["d"].unique()
    workday_days = df[df["is_holiday"] == 0]["d"].unique()
    stats["holiday_days_count"] = int(len(holiday_days))
    stats["workday_days_count"] = int(len(workday_days))
    stats["holiday_days"] = sorted(int(d) for d in holiday_days)
    stats["holiday_daily_mean"] = float(df[df["is_holiday"] == 1].groupby("d")["uid"].nunique().mean())
    stats["workday_daily_mean"] = float(df[df["is_holiday"] == 0].groupby("d")["uid"].nunique().mean())

    # Spatial coverage
    occupied = df.groupby(["x", "y"]).size()
    stats["occupied_cells"] = int(len(occupied))
    stats["total_cells"] = GRID_SIZE * GRID_SIZE
    stats["spatial_coverage_pct"] = round(100 * len(occupied) / (GRID_SIZE * GRID_SIZE), 2)

    # Top 10 most visited cells
    top10 = occupied.sort_values(ascending=False).head(10).reset_index()
    top10.columns = ["x", "y", "visits"]
    top10["lat"] = BBOX["south"] + top10["x"] * _LAT_STEP
    top10["lon"] = BBOX["west"] + top10["y"] * _LON_STEP
    stats["top10_cells"] = top10.to_dict("records")

    _log(f"  users={stats['total_users']:,}  holiday_days={stats['holiday_days_count']}  workdays={stats['workday_days_count']}")
    return stats


# ── 3. Hourly / time-of-day patterns ──────────────────────────────────────────
def step3_time_patterns(df: pd.DataFrame) -> dict:
    _log("Computing time-of-day patterns …")
    stats: dict = {}

    # Average active users per time slot
    hourly = df.groupby("t")["uid"].nunique().reset_index()
    hourly.columns = ["t", "unique_users"]

    # Separate weekday vs holiday
    workday = df[df["is_holiday"] == 0].groupby("t")["uid"].nunique()
    holiday = df[df["is_holiday"] == 1].groupby("t")["uid"].nunique()

    peak_t = int(hourly.loc[hourly["unique_users"].idxmax(), "t"])
    trough_t = int(hourly.loc[hourly["unique_users"].idxmin(), "t"])
    stats["peak_time_slot"] = peak_t
    stats["peak_time_hhmm"] = f"{(peak_t * 30) // 60:02d}:{(peak_t * 30) % 60:02d}"
    stats["trough_time_slot"] = trough_t
    stats["trough_time_hhmm"] = f"{(trough_t * 30) // 60:02d}:{(trough_t * 30) % 60:02d}"

    # Save plot
    fig, ax = plt.subplots(figsize=(14, 4))
    times = [f"{(t * 30) // 60:02d}:{(t * 30) % 60:02d}" for t in range(48)]
    ax.plot(range(48), workday.values, label="Weekday", color="steelblue", linewidth=1.5)
    ax.plot(range(48), holiday.values, label="Holiday/Weekend", color="tomato", linewidth=1.5)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([times[t] for t in range(0, 48, 4)], rotation=45, fontsize=8)
    ax.set_title("Nagoya - Active Users by Time Slot (Weekday vs Holiday)")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Unique Users")
    ax.legend()
    plt.tight_layout()
    path = FIG_DIR / "time_of_day_pattern.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved: {path}")

    # Peak hours analysis (morning 6-9, noon 11-13, evening 17-20)
    morning_peak = workday.iloc[12:18].mean()  # t=12~17 = 6:00~8:30
    noon_peak = workday.iloc[22:26].mean()     # t=22~25 = 11:00~12:30
    evening_peak = workday.iloc[34:40].mean()  # t=34~39 = 17:00~19:30
    stats["morning_peak_users"] = float(morning_peak)
    stats["noon_peak_users"] = float(noon_peak)
    stats["evening_peak_users"] = float(evening_peak)

    _log(f"  peak_slot={peak_t} ({stats['peak_time_hhmm']})  trough_slot={trough_t} ({stats['trough_time_hhmm']})")
    return stats


# ── 4. User stability analysis ────────────────────────────────────────────────
def step4_user_stability(train_df: pd.DataFrame) -> dict:
    _log("Computing user stability (std per uid per time slot) …")
    stats: dict = {}

    # Standard deviation of x and y for each user across days, per time slot
    workday = train_df[train_df["is_holiday"] == 0]
    stability = workday.groupby(["uid", "t"])[["x", "y"]].std().reset_index()
    user_std = stability.groupby("uid")[["x", "y"]].mean().reset_index()
    user_std["std_mean"] = (user_std["x"] + user_std["y"]) / 2

    std_bins = [0, 1, 2, 3, 5, 10, 20, 999]
    std_labels = ["<1", "1-2", "2-3", "3-5", "5-10", "10-20", ">=20"]
    user_std["std_group"] = pd.cut(user_std["std_mean"], bins=std_bins, labels=std_labels)
    dist = user_std["std_group"].value_counts().sort_index()

    total = len(user_std)
    stats["stability_total_users"] = total
    stats["stability_distribution"] = {
        label: {"count": int(cnt), "pct": round(100 * cnt / total, 1)}
        for label, cnt in dist.items()
    }

    # Very stable (std < 5) = highly predictable
    very_stable = (user_std["std_mean"] < 5).sum()
    stats["very_stable_users"] = int(very_stable)
    stats["very_stable_pct"] = round(100 * very_stable / total, 1)
    stats["stable_users_std10"] = int((user_std["std_mean"] < 10).sum())
    stats["stable_users_std10_pct"] = round(100 * (user_std["std_mean"] < 10).sum() / total, 1)

    # Plot std distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(dist.index.astype(str), dist.values, color="steelblue")
    ax.set_title("Nagoya - User Location Stability (Weekday, Avg Std per Time Slot)")
    ax.set_xlabel("Avg Std (grid cells)")
    ax.set_ylabel("Number of Users")
    for i, (label, cnt) in enumerate(dist.items()):
        ax.text(i, cnt + 100, f"{100*cnt/total:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    path = FIG_DIR / "user_stability_std.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved: {path}")

    _log(f"  very_stable (std<5)={very_stable:,} ({stats['very_stable_pct']}%)  std<10={stats['stable_users_std10']:,} ({stats['stable_users_std10_pct']}%)")
    return stats


# ── 5. HDBSCAN clustering ─────────────────────────────────────────────────────
def step5_hdbscan(train_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    _log("Running HDBSCAN spatial clustering …")
    stats: dict = {}

    density = compute_grid_density(train_df)
    cluster_map = run_hdbscan(density, min_cluster_size=10, min_samples=5)

    n_noise = int((cluster_map["cluster_id"] == -1).sum())
    n_clusters = int(cluster_map[cluster_map["cluster_id"] >= 0]["cluster_id"].nunique())
    stats["hdbscan_total_cells"] = int(len(cluster_map))
    stats["hdbscan_n_clusters"] = n_clusters
    stats["hdbscan_noise_cells"] = n_noise
    stats["hdbscan_noise_pct"] = round(100 * n_noise / len(cluster_map), 1)

    # Cluster sizes
    cluster_sizes = (
        cluster_map[cluster_map["cluster_id"] >= 0]
        .groupby("cluster_id").size()
        .sort_values(ascending=False)
    )
    stats["top10_clusters"] = [
        {"cluster_id": int(cid), "n_cells": int(sz)}
        for cid, sz in cluster_sizes.head(10).items()
    ]

    # Proximity to Nagoya Station
    ns_x, ns_y = NAGOYA_STATION_GRID["x"], NAGOYA_STATION_GRID["y"]
    cluster_map["dist_to_nagoya_st"] = np.sqrt(
        (cluster_map["x"] - ns_x) ** 2 + (cluster_map["y"] - ns_y) ** 2
    )
    nearest = cluster_map.nsmallest(5, "dist_to_nagoya_st")[["x", "y", "cluster_id", "count", "dist_to_nagoya_st"]]
    stats["nagoya_station_grid"] = NAGOYA_STATION_GRID
    stats["nagoya_station_nearby_clusters"] = nearest.to_dict("records")

    # Plot cluster map
    canvas = np.full((GRID_SIZE, GRID_SIZE), -2, dtype=int)
    for _, row in cluster_map.iterrows():
        xi, yi = int(row["x"]), int(row["y"])
        if 0 <= xi < GRID_SIZE and 0 <= yi < GRID_SIZE:
            canvas[yi, xi] = int(row["cluster_id"])

    fig, ax = plt.subplots(figsize=(10, 10))
    masked_noise = np.ma.masked_where(canvas != -1, canvas)
    masked_valid = np.ma.masked_where(canvas < 0, canvas)
    ax.imshow(canvas.T, origin="lower", cmap="tab20", interpolation="nearest", alpha=0.6)
    ax.set_title(f"Nagoya - HDBSCAN Clusters ({n_clusters} clusters, noise={n_noise} cells)")
    ax.set_xlabel("Grid Y (West -> East)")
    ax.set_ylabel("Grid X (South -> North)")
    # Mark Nagoya Station
    ax.plot(ns_y, ns_x, "r*", markersize=15, label=f"Nagoya Station (x={ns_x},y={ns_y})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = FIG_DIR / "hdbscan_clusters.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved: {path}")

    _log(f"  clusters={n_clusters}  noise_cells={n_noise} ({stats['hdbscan_noise_pct']}%)")
    return stats, cluster_map


# ── 6. Nagoya Station area analysis ───────────────────────────────────────────
def step6_nagoya_station(df: pd.DataFrame, cluster_map: pd.DataFrame) -> dict:
    _log("Analysing Nagoya Station area …")
    stats: dict = {}
    ns_x, ns_y = NAGOYA_STATION_GRID["x"], NAGOYA_STATION_GRID["y"]

    # Activity in a 5-cell radius around Nagoya Station
    radius = 5
    area = df[
        (df["x"].between(ns_x - radius, ns_x + radius)) &
        (df["y"].between(ns_y - radius, ns_y + radius))
    ]
    stats["station_area_visits"] = int(len(area))
    stats["station_area_unique_users"] = int(area["uid"].nunique())
    stats["station_area_pct_of_total"] = round(100 * len(area) / len(df), 2)

    # Time-of-day pattern at station area
    area_hourly = area.groupby("t")["uid"].nunique()
    peak_t = int(area_hourly.idxmax())
    stats["station_peak_slot"] = peak_t
    stats["station_peak_hhmm"] = f"{(peak_t * 30) // 60:02d}:{(peak_t * 30) % 60:02d}"

    # Weekday vs holiday activity
    wday_mean = area[area["is_holiday"] == 0].groupby("d")["uid"].nunique().mean()
    hday_mean = area[area["is_holiday"] == 1].groupby("d")["uid"].nunique().mean()
    stats["station_weekday_daily_users"] = float(wday_mean)
    stats["station_holiday_daily_users"] = float(hday_mean)
    stats["station_weekday_holiday_ratio"] = round(float(wday_mean / hday_mean), 2) if hday_mean > 0 else 0

    # Cluster ID of Nagoya Station grid cell
    ns_cell = cluster_map[
        (cluster_map["x"].between(ns_x - 2, ns_x + 2)) &
        (cluster_map["y"].between(ns_y - 2, ns_y + 2))
    ].sort_values("count", ascending=False)
    if not ns_cell.empty:
        stats["station_cluster_id"] = int(ns_cell.iloc[0]["cluster_id"])
        stats["station_cell_visits"] = int(ns_cell.iloc[0]["count"])

    # Plot station area time pattern
    area_work = area[area["is_holiday"] == 0].groupby("t")["uid"].nunique()
    area_holi = area[area["is_holiday"] == 1].groupby("t")["uid"].nunique()
    times = [f"{(t * 30) // 60:02d}:{(t * 30) % 60:02d}" for t in range(48)]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(range(48), area_work.reindex(range(48), fill_value=0).values,
            label="Weekday", color="steelblue", linewidth=1.5)
    ax.plot(range(48), area_holi.reindex(range(48), fill_value=0).values,
            label="Holiday/Weekend", color="tomato", linewidth=1.5)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([times[t] for t in range(0, 48, 4)], rotation=45, fontsize=8)
    ax.set_title(f"Nagoya Station Area (x={ns_x}±{radius}, y={ns_y}±{radius}) - Active Users by Time Slot")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Unique Users in Area")
    ax.legend()
    plt.tight_layout()
    path = FIG_DIR / "nagoya_station_time_pattern.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved: {path}")

    _log(f"  station area: {stats['station_area_unique_users']:,} unique users, "
         f"weekday/holiday ratio={stats['station_weekday_holiday_ratio']}")
    return stats


# ── 7. EDA figures ─────────────────────────────────────────────────────────────
def step7_eda_figures(df: pd.DataFrame, train_df: pd.DataFrame) -> None:
    _log("Generating EDA figures …")
    plot_daily_active_users(df, save_path=str(FIG_DIR / "daily_active_users.png"))
    plot_unique_users_map(df, save_path=str(FIG_DIR / "unique_users_map.png"))
    plot_spatial_heatmap(df, save_path=str(FIG_DIR / "spatial_heatmap.png"))
    # Trajectory map is slow — use 5000 random users
    plot_trajectory_map(
        train_df, n_users=5000,
        save_path=str(FIG_DIR / "trajectory_map_morning.png"),
        time_slice=(16, 20),  # 8:00-10:00
    )
    plot_trajectory_map(
        train_df, n_users=5000,
        save_path=str(FIG_DIR / "trajectory_map_evening.png"),
        time_slice=(34, 38),  # 17:00-19:00
    )


# ── Main ────────────────────────────────────────────────────────────────────────
def main(city_path: str) -> None:
    all_stats: dict = {}

    df, train_df, test_df = step1_load(city_path)
    all_stats["basic"] = step2_basic_stats(df)
    all_stats["time"] = step3_time_patterns(df)
    all_stats["stability"] = step4_user_stability(train_df)
    all_stats["hdbscan"], cluster_map = step5_hdbscan(train_df)
    all_stats["nagoya_station"] = step6_nagoya_station(df, cluster_map)
    step7_eda_figures(df, train_df)

    # Save all stats
    out_path = ROOT / "output" / "stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    _log(f"Stats saved to {out_path}")
    _log("Analysis complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-path", default="raw_data/nagoya_challengedata.csv")
    args = parser.parse_args()
    main(args.city_path)
