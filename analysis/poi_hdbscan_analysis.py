"""
Deep-dive analysis:
  1. 4 POIs: 名古屋大学, 名古屋城, 大須商店街, 金山
  2. HDBSCAN re-analysis with better parameters and richer visualisation
"""
from __future__ import annotations

import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from data.preprocessing import BBOX, _LAT_STEP, _LON_STEP, GRID_SIZE

TIMES = [f"{(t*30)//60:02d}:{(t*30)%60:02d}" for t in range(48)]


def _log(msg: str) -> None:
    print(f"[analysis] {msg}", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    _log("Loading parquet …")
    df = pd.read_parquet(ROOT / "output" / "nagoya_clean.parquet")
    _log(f"  {len(df):,} rows, {df['uid'].nunique():,} users")
    return df


def area_stats(df: pd.DataFrame, cx: int, cy: int, radius: int = 3) -> dict:
    mask = (df["x"].between(cx - radius, cx + radius) &
            df["y"].between(cy - radius, cy + radius))
    sub = df[mask]
    if sub.empty:
        return {}
    wday = sub[sub["is_holiday"] == 0].groupby("d")["uid"].nunique()
    hday = sub[sub["is_holiday"] == 1].groupby("d")["uid"].nunique()
    return {
        "total_visits": int(len(sub)),
        "unique_users": int(sub["uid"].nunique()),
        "pct_of_total": round(100 * len(sub) / len(df), 2),
        "weekday_daily_mean": float(wday.mean()) if len(wday) else 0,
        "holiday_daily_mean": float(hday.mean()) if len(hday) else 0,
        "weekday_holiday_ratio": round(float(wday.mean() / hday.mean()), 2) if len(hday) and hday.mean() > 0 else 0,
        "peak_slot": int(sub.groupby("t")["uid"].nunique().idxmax()),
    }


def plot_poi_time(df: pd.DataFrame, cx: int, cy: int, radius: int,
                  poi_name: str, save_path: Path) -> None:
    mask = (df["x"].between(cx - radius, cx + radius) &
            df["y"].between(cy - radius, cy + radius))
    sub = df[mask]
    wday = sub[sub["is_holiday"] == 0].groupby("t")["uid"].nunique().reindex(range(48), fill_value=0)
    hday = sub[sub["is_holiday"] == 1].groupby("t")["uid"].nunique().reindex(range(48), fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(range(48), wday.values, label="Weekday", color="steelblue", linewidth=2)
    ax.plot(range(48), hday.values, label="Holiday/Weekend", color="tomato", linewidth=2)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([TIMES[t] for t in range(0, 48, 4)], rotation=45, fontsize=8)
    ax.set_title(f"{poi_name}  –  Active Users by Time Slot (grid ±{radius} cells)")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Unique Users in Area")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    _log(f"  saved: {save_path.name}")


# ── POI analysis ───────────────────────────────────────────────────────────────
POIS = {
    "nagoya_univ":   {"name": "Nagoya University", "cx": 131, "cy": 87, "r": 3},
    "nagoya_castle": {"name": "Nagoya Castle",     "cx": 136, "cy": 80, "r": 3},
    "osu":           {"name": "Osu Shopping St.",  "cx": 131, "cy": 80, "r": 3},
    "kanayama":      {"name": "Kanayama",          "cx": 129, "cy": 81, "r": 3},
}


def run_poi_analysis(df: pd.DataFrame) -> dict:
    _log("=== POI deep-dive analysis ===")
    results = {}
    for key, poi in POIS.items():
        _log(f"  → {poi['name']}")
        stats = area_stats(df, poi["cx"], poi["cy"], poi["r"])
        stats["peak_hhmm"] = TIMES[stats["peak_slot"]]
        results[key] = {**poi, **stats}
        plot_poi_time(
            df, poi["cx"], poi["cy"], poi["r"], poi["name"],
            FIG_DIR / f"poi_{key}_time.png",
        )
    # Combined 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, (key, poi) in zip(axes.flat, POIS.items()):
        mask = (df["x"].between(poi["cx"] - poi["r"], poi["cx"] + poi["r"]) &
                df["y"].between(poi["cy"] - poi["r"], poi["cy"] + poi["r"]))
        sub = df[mask]
        wday = sub[sub["is_holiday"] == 0].groupby("t")["uid"].nunique().reindex(range(48), fill_value=0)
        hday = sub[sub["is_holiday"] == 1].groupby("t")["uid"].nunique().reindex(range(48), fill_value=0)
        ax.plot(range(48), wday.values, label="Weekday", color="steelblue", linewidth=1.5)
        ax.plot(range(48), hday.values, label="Holiday", color="tomato", linewidth=1.5)
        ratio = results[key].get("weekday_holiday_ratio", 0)
        ax.set_title(f"{poi['name']}\nWkday/Holiday ratio = {ratio}×", fontsize=10)
        ax.set_xticks(range(0, 48, 8))
        ax.set_xticklabels([TIMES[t] for t in range(0, 48, 8)], fontsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("Nagoya – 4 POI Comparison: Active Users by Time Slot", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "poi_4panel_comparison.png", dpi=150)
    plt.close()
    _log(f"  saved: poi_4panel_comparison.png")
    return results


# ── HDBSCAN deep analysis ──────────────────────────────────────────────────────
def run_hdbscan_analysis(df: pd.DataFrame) -> dict:
    _log("=== HDBSCAN deep analysis ===")
    from analysis.clustering import compute_grid_density, run_hdbscan

    train = df[df["d"] <= 60]
    density = compute_grid_density(train)
    total_cells = len(density)
    _log(f"  Density cells: {total_cells:,}  visit range: {density['count'].min()}~{density['count'].max():,}")

    # ── run 1: default (all cells, min_size=10) ─────────────────────────────
    cm_all = run_hdbscan(density, min_cluster_size=10, min_samples=5)
    n1 = cm_all[cm_all["cluster_id"] >= 0]["cluster_id"].nunique()
    noise1 = int((cm_all["cluster_id"] == -1).sum())
    _log(f"  Run1 (all cells, min_size=10): {n1} clusters, noise={noise1}")

    # ── run 2: top-density cells only (count >= 75th pct), min_size=30 ────
    p75 = int(density["count"].quantile(0.75))
    dense = density[density["count"] >= p75].copy()
    cm_dense = run_hdbscan(dense, min_cluster_size=30, min_samples=15)
    n2 = cm_dense[cm_dense["cluster_id"] >= 0]["cluster_id"].nunique()
    noise2 = int((cm_dense["cluster_id"] == -1).sum())
    _log(f"  Run2 (top-25% density, min_size=30): {n2} clusters, noise={noise2}  (threshold≥{p75:,})")

    # ── run 3: top-10% cells, min_size=20 ──────────────────────────────────
    p90 = int(density["count"].quantile(0.90))
    hotspot = density[density["count"] >= p90].copy()
    cm_hot = run_hdbscan(hotspot, min_cluster_size=20, min_samples=10)
    n3 = cm_hot[cm_hot["cluster_id"] >= 0]["cluster_id"].nunique()
    noise3 = int((cm_hot["cluster_id"] == -1).sum())
    _log(f"  Run3 (top-10% density, min_size=20): {n3} clusters, noise={noise3}  (threshold≥{p90:,})")

    # ── enrichment: per-cluster visit stats (Run3) ─────────────────────────
    cluster_stats = []
    for cid, grp in cm_hot[cm_hot["cluster_id"] >= 0].groupby("cluster_id"):
        cx = float(grp["x"].mean())
        cy = float(grp["y"].mean())
        lat = BBOX["south"] + cx * _LAT_STEP
        lon = BBOX["west"]  + cy * _LON_STEP
        cluster_stats.append({
            "cluster_id": int(cid),
            "n_cells": int(len(grp)),
            "total_visits": int(grp["count"].sum()),
            "center_x": round(cx, 1), "center_y": round(cy, 1),
            "center_lat": round(lat, 4), "center_lon": round(lon, 4),
        })
    cluster_stats.sort(key=lambda r: -r["total_visits"])

    # assign POI labels by proximity
    poi_coords = [
        ("Nagoya Sta.", 133, 76), ("Sakae", 133, 81), ("Castle", 136, 80),
        ("Osu", 131, 80), ("Kanayama", 129, 81), ("Dome", 136, 89),
        ("Atsuta", 124, 81), ("Nagoya Univ.", 131, 87), ("Kakuozan", 133, 86),
    ]
    for cs in cluster_stats:
        dists = [(n, np.hypot(cs["center_x"]-px, cs["center_y"]-py))
                 for n, px, py in poi_coords]
        nearest_poi, nearest_dist = min(dists, key=lambda x: x[1])
        cs["nearest_poi"] = nearest_poi
        cs["dist_to_poi"] = round(float(nearest_dist), 1)

    _log(f"  Top clusters (Run3):")
    for cs in cluster_stats[:8]:
        _log(f"    Cluster {cs['cluster_id']:2d}  cells={cs['n_cells']:4d}  "
             f"visits={cs['total_visits']:>9,}  near={cs['nearest_poi']} ({cs['dist_to_poi']} cells)")

    # ── density distribution stats ─────────────────────────────────────────
    bins = [0, 100, 500, 2000, 10000, 50000, 999_999_999]
    labels = ["<100", "100-500", "500-2k", "2k-10k", "10k-50k", "≥50k"]
    density["visit_bin"] = pd.cut(density["count"], bins=bins, labels=labels)
    bin_dist = density.groupby("visit_bin", observed=True).agg(
        n_cells=("count", "size"), total_visits=("count", "sum")
    ).reset_index()

    # ── figures ────────────────────────────────────────────────────────────
    _plot_hdbscan_3panel(cm_all, cm_dense, cm_hot, p75, p90, n1, n2, n3)
    _plot_density_distribution(density, bin_dist)
    _plot_cluster_map_annotated(cm_hot, cluster_stats, poi_coords)
    _plot_cluster_visit_bar(cluster_stats)

    return {
        "run1": {"n_clusters": n1, "noise": noise1},
        "run2": {"n_clusters": n2, "noise": noise2, "threshold": int(p75)},
        "run3": {"n_clusters": n3, "noise": noise3, "threshold": int(p90)},
        "top_clusters": cluster_stats[:12],
        "density_bins": bin_dist.to_dict("records"),
    }


def _plot_hdbscan_3panel(cm_all, cm_dense, cm_hot, p75, p90, n1, n2, n3):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    configs = [
        (cm_all,   "All cells (min_size=10)",            f"{n1} clusters"),
        (cm_dense, f"Top-25% density (≥{p75:,}, min30)", f"{n2} clusters"),
        (cm_hot,   f"Top-10% density (≥{p90:,}, min20)", f"{n3} clusters"),
    ]
    for ax, (cm, subtitle, legend) in zip(axes, configs):
        canvas = np.full((GRID_SIZE, GRID_SIZE), -2, dtype=float)
        for _, row in cm.iterrows():
            xi, yi = int(row["x"]), int(row["y"])
            if 0 <= xi < GRID_SIZE and 0 <= yi < GRID_SIZE:
                canvas[yi, xi] = float(row["cluster_id"])
        ax.imshow(canvas.T, origin="lower", cmap="tab20", vmin=-1, vmax=max(1, n1),
                  interpolation="nearest", alpha=0.8)
        # mark Nagoya Station
        ax.plot(76, 133, "r*", markersize=12, label="Nagoya Sta.")
        ax.plot(81, 133, "w^", markersize=8, label="Sakae")
        ax.plot(80, 136, "ys", markersize=8, label="Castle")
        ax.set_title(f"{subtitle}\n→ {legend}", fontsize=10)
        ax.set_xlabel("y (West→East)")
        ax.set_ylabel("x (South→North)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("HDBSCAN Parameter Sensitivity – Nagoya Spatial Clustering", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_3panel.png", dpi=150)
    plt.close()
    _log("  saved: hdbscan_3panel.png")


def _plot_density_distribution(density: pd.DataFrame, bin_dist: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: histogram of cell visit counts (log scale)
    axes[0].hist(np.log10(density["count"].clip(lower=1)), bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("log10(visit count per cell)")
    axes[0].set_ylabel("Number of Cells")
    axes[0].set_title("Distribution of Cell Visit Counts (log10 scale)")
    axes[0].axvline(np.log10(density["count"].quantile(0.75)), color="orange", linestyle="--", label="75th pct")
    axes[0].axvline(np.log10(density["count"].quantile(0.90)), color="red", linestyle="--", label="90th pct")
    axes[0].legend()
    # right: bar chart of bin counts
    colors = ["#d0d0d0","#a0c4e8","#5fa8d3","#2d7db3","#1a4f7a","#0a1f3a"]
    bars = axes[1].bar(bin_dist["visit_bin"].astype(str), bin_dist["n_cells"], color=colors[:len(bin_dist)])
    axes[1].set_xlabel("Visit Count Range")
    axes[1].set_ylabel("Number of Grid Cells")
    axes[1].set_title("Grid Cells by Visit Count Tier")
    for bar, row in zip(bars, bin_dist.itertuples()):
        pct = 100 * row.n_cells / density["count"].count()
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f"{pct:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_density_distribution.png", dpi=150)
    plt.close()
    _log("  saved: hdbscan_density_distribution.png")


def _plot_cluster_map_annotated(cm_hot: pd.DataFrame, cluster_stats: list, poi_coords: list):
    canvas = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    for _, row in cm_hot.iterrows():
        xi, yi = int(row["x"]), int(row["y"])
        if 0 <= xi < GRID_SIZE and 0 <= yi < GRID_SIZE:
            canvas[yi, xi] = float(row["cluster_id"]) if row["cluster_id"] >= 0 else -1

    fig, ax = plt.subplots(figsize=(12, 12))
    cmap = plt.get_cmap("tab20")
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(canvas).T, origin="lower", cmap=cmap,
                   interpolation="nearest", alpha=0.85)
    # annotate top clusters
    for cs in cluster_stats[:10]:
        ax.text(cs["center_y"], cs["center_x"], f"C{cs['cluster_id']}",
                color="black", fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    # mark known POIs
    colors_poi = ["red","white","yellow","lime","cyan","orange","magenta","pink","lightblue"]
    markers = ["*","^","s","D","o","P","X","v","h"]
    for (name, px, py), c, m in zip(poi_coords, colors_poi, markers):
        ax.plot(py, px, m, color=c, markersize=12, label=name,
                markeredgecolor="black", markeredgewidth=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("HDBSCAN – Top-10% Density Cells\nAnnotated Clusters & Major POIs", fontsize=12)
    ax.set_xlabel("Grid Y (West → East)")
    ax.set_ylabel("Grid X (South → North)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_annotated.png", dpi=150)
    plt.close()
    _log("  saved: hdbscan_annotated.png")


def _plot_cluster_visit_bar(cluster_stats: list):
    top = cluster_stats[:10]
    labels = [f"C{cs['cluster_id']}\n({cs['nearest_poi']})" for cs in top]
    visits = [cs["total_visits"] for cs in top]
    cells  = [cs["n_cells"] for cs in top]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(labels[::-1], [v/1000 for v in visits[::-1]], color="steelblue")
    axes[0].set_xlabel("Total Visits (×1,000)")
    axes[0].set_title("HDBSCAN Cluster – Total Visits (Training Set)")
    axes[1].barh(labels[::-1], cells[::-1], color="coral")
    axes[1].set_xlabel("Number of Grid Cells")
    axes[1].set_title("HDBSCAN Cluster – Spatial Extent (Cell Count)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_cluster_bars.png", dpi=150)
    plt.close()
    _log("  saved: hdbscan_cluster_bars.png")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    poi_results   = run_poi_analysis(df)
    hdbscan_stats = run_hdbscan_analysis(df)

    out = {"poi": poi_results, "hdbscan": hdbscan_stats}
    out_path = ROOT / "output" / "poi_hdbscan_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log(f"Saved {out_path}")


if __name__ == "__main__":
    main()
