"""
HDBSCAN deep-dive for report Section 5 expansion:
  5.1.1 core-distance contrast (algorithm rationale, concrete numbers)
  5.3   empirical comparison: HDBSCAN vs. K-Means vs. DBSCAN (same input, same scale)
  5.4   systematic parameter-sensitivity sweeps (min_cluster_size sweep / density-threshold sweep)
  5.6   temporal rhythm of top HDBSCAN clusters (weekday vs. holiday, peak time)
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from data.preprocessing import BBOX, _LAT_STEP, _LON_STEP, GRID_SIZE
from analysis.clustering import compute_grid_density, run_hdbscan

TIMES = [f"{(t*30)//60:02d}:{(t*30)%60:02d}" for t in range(48)]
NAGOYA_STATION = (133, 76)


def _log(msg: str) -> None:
    print(f"[deepdive] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    _log("Loading parquet …")
    df = pd.read_parquet(ROOT / "output" / "nagoya_clean.parquet")
    _log(f"  {len(df):,} rows, {df['uid'].nunique():,} users")
    return df


# ── 5.1.1 core-distance contrast ───────────────────────────────────────────────
def core_distance_contrast(density: pd.DataFrame, k: int = 5) -> dict:
    _log("=== 5.1.1 core-distance contrast ===")
    from sklearn.neighbors import NearestNeighbors

    coords = density[["x", "y"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dist, _ = nn.kneighbors(coords)
    core_dist = dist[:, k]  # distance to k-th neighbour = HDBSCAN core distance
    density = density.copy()
    density["core_dist"] = core_dist

    # Nagoya station cell core distance
    st_row = density[(density["x"] == NAGOYA_STATION[0]) & (density["y"] == NAGOYA_STATION[1])]
    st_core = float(st_row["core_dist"].iloc[0]) if len(st_row) else float("nan")

    # rural / sparse cells: bottom-quartile visit-count cells
    q25 = density["count"].quantile(0.25)
    rural = density[density["count"] <= q25]
    rural_core_median = float(rural["core_dist"].median())
    rural_core_mean = float(rural["core_dist"].mean())

    ratio = rural_core_median / st_core if st_core > 0 else float("nan")
    result = {
        "k": k,
        "station_grid": NAGOYA_STATION,
        "station_core_dist": round(st_core, 4),
        "station_visit_count": int(st_row["count"].iloc[0]) if len(st_row) else None,
        "rural_cell_count": int(len(rural)),
        "rural_core_dist_median": round(rural_core_median, 4),
        "rural_core_dist_mean": round(rural_core_mean, 4),
        "ratio_rural_to_station": round(ratio, 1),
        "global_core_dist_percentiles": {
            p: round(float(np.percentile(core_dist, p)), 4) for p in (10, 25, 50, 75, 90, 99)
        },
    }
    _log(f"  station core_dist={st_core:.3f}  rural median={rural_core_median:.3f}  ratio={ratio:.1f}x")
    return result


# ── 5.3 empirical comparison: HDBSCAN vs K-Means vs DBSCAN ─────────────────────
def find_dbscan_eps(coords: np.ndarray, k: int = 5) -> float:
    """Pick eps at the 'knee' of the sorted k-distance curve (max curvature)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dist, _ = nn.kneighbors(coords)
    kdist = np.sort(dist[:, k])
    # normalise to [0,1] on both axes, find point with max distance from the
    # straight line connecting the curve endpoints (classic knee heuristic)
    n = len(kdist)
    x = np.arange(n) / (n - 1)
    y = (kdist - kdist.min()) / (kdist.max() - kdist.min() + 1e-12)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    pts = np.column_stack([x, y]) - p1
    proj_len = pts @ line_vec_norm
    proj = np.outer(proj_len, line_vec_norm)
    dist_to_line = np.linalg.norm(pts - proj, axis=1)
    knee_idx = int(np.argmax(dist_to_line))
    eps = float(kdist[knee_idx])
    return eps, kdist


def run_comparison(density: pd.DataFrame) -> dict:
    _log("=== 5.3 empirical comparison: HDBSCAN vs K-Means vs DBSCAN ===")
    from sklearn.cluster import KMeans, DBSCAN

    coords = density[["x", "y"]].to_numpy(dtype=float)

    # HDBSCAN (= Run1 setting, the report's primary configuration)
    cm_hdb = run_hdbscan(density, min_cluster_size=10, min_samples=5)
    n_hdb = cm_hdb[cm_hdb["cluster_id"] >= 0]["cluster_id"].nunique()
    noise_hdb = int((cm_hdb["cluster_id"] == -1).sum())

    # K-Means with k matched to HDBSCAN's cluster count
    k = max(2, n_hdb)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords)
    cm_km = density.copy()
    cm_km["cluster_id"] = km.labels_
    sizes_km = cm_km.groupby("cluster_id").size()

    # DBSCAN with eps chosen via k-distance knee heuristic
    eps, kdist_curve = find_dbscan_eps(coords, k=5)
    db = DBSCAN(eps=eps, min_samples=5).fit(coords)
    cm_db = density.copy()
    cm_db["cluster_id"] = db.labels_
    n_db = cm_db[cm_db["cluster_id"] >= 0]["cluster_id"].nunique()
    noise_db = int((cm_db["cluster_id"] == -1).sum())
    sizes_db = cm_db[cm_db["cluster_id"] >= 0].groupby("cluster_id").size().sort_values(ascending=False)

    result = {
        "hdbscan": {
            "params": "min_cluster_size=10, min_samples=5",
            "n_clusters": int(n_hdb), "noise": noise_hdb,
            "noise_pct": round(100 * noise_hdb / len(density), 2),
            "largest_cluster_pct": round(
                100 * cm_hdb[cm_hdb["cluster_id"] >= 0].groupby("cluster_id").size().max() / len(density), 1),
        },
        "kmeans": {
            "params": f"k={k} (matched to HDBSCAN cluster count)",
            "n_clusters": int(k), "noise": 0, "noise_pct": 0.0,
            "cluster_size_min": int(sizes_km.min()), "cluster_size_max": int(sizes_km.max()),
            "cluster_size_std": round(float(sizes_km.std()), 1),
        },
        "dbscan": {
            "params": f"eps={eps:.3f} (k-distance knee, k=5), min_samples=5",
            "eps": round(eps, 4),
            "n_clusters": int(n_db), "noise": noise_db,
            "noise_pct": round(100 * noise_db / len(density), 2),
            "largest_cluster_pct": round(100 * sizes_db.max() / len(density), 1) if len(sizes_db) else None,
        },
    }
    _log(f"  HDBSCAN: {n_hdb} clusters, noise={noise_hdb} ({result['hdbscan']['noise_pct']}%)")
    _log(f"  K-Means(k={k}): cluster size std={result['kmeans']['cluster_size_std']} "
         f"(min={result['kmeans']['cluster_size_min']}, max={result['kmeans']['cluster_size_max']})")
    _log(f"  DBSCAN(eps={eps:.3f}): {n_db} clusters, noise={noise_db} ({result['dbscan']['noise_pct']}%), "
         f"largest cluster={result['dbscan']['largest_cluster_pct']}%")

    _plot_comparison(cm_hdb, cm_km, cm_db, result)
    _plot_kdistance(kdist_curve, eps)
    return result


def _canvas(cm: pd.DataFrame) -> np.ndarray:
    canvas = np.full((GRID_SIZE, GRID_SIZE), -2, dtype=float)
    for _, row in cm.iterrows():
        xi, yi = int(row["x"]), int(row["y"])
        if 0 <= xi < GRID_SIZE and 0 <= yi < GRID_SIZE:
            canvas[yi, xi] = float(row["cluster_id"])
    return canvas


def _plot_comparison(cm_hdb, cm_km, cm_db, stats: dict):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    configs = [
        (cm_hdb, "HDBSCAN (min_size=10)",
         f"{stats['hdbscan']['n_clusters']} clusters, noise={stats['hdbscan']['noise_pct']}%"),
        (cm_km, f"K-Means (k={stats['kmeans']['n_clusters']})",
         f"size std={stats['kmeans']['cluster_size_std']} (forced equal-ish)"),
        (cm_db, f"DBSCAN (eps={stats['dbscan']['eps']})",
         f"{stats['dbscan']['n_clusters']} clusters, noise={stats['dbscan']['noise_pct']}%"),
    ]
    for ax, (cm, subtitle, legend) in zip(axes, configs):
        canvas = _canvas(cm)
        vmax = max(int(cm["cluster_id"].max()), 1)
        ax.imshow(canvas.T, origin="lower", cmap="tab20", vmin=-1, vmax=vmax,
                  interpolation="nearest", alpha=0.85)
        ax.plot(NAGOYA_STATION[1], NAGOYA_STATION[0], "r*", markersize=12, label="Nagoya Sta.")
        ax.set_title(f"{subtitle}\n{legend}", fontsize=10)
        ax.set_xlabel("y (West→East)")
        ax.set_ylabel("x (South→North)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Empirical Comparison – HDBSCAN vs. K-Means vs. DBSCAN (same 33,284 cells)", fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_vs_kmeans_dbscan.png", dpi=150, bbox_inches="tight")
    plt.close()
    _log("  saved: hdbscan_vs_kmeans_dbscan.png")


def _plot_kdistance(kdist_curve: np.ndarray, eps: float):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(kdist_curve, color="steelblue", linewidth=1.5)
    ax.axhline(eps, color="red", linestyle="--", label=f"knee eps = {eps:.3f}")
    ax.set_xlabel("Cells sorted by 5-NN distance")
    ax.set_ylabel("5-NN distance (grid cells)")
    ax.set_title("DBSCAN eps Selection – k-distance Knee Point")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dbscan_kdistance_knee.png", dpi=150)
    plt.close()
    _log("  saved: dbscan_kdistance_knee.png")


# ── 5.4 systematic parameter-sensitivity sweeps ────────────────────────────────
def sweep_min_cluster_size(density: pd.DataFrame,
                           values=(5, 10, 20, 30, 50, 80, 100)) -> list[dict]:
    _log("=== 5.4a sweep: min_cluster_size (fixed input = all 33,284 cells) ===")
    rows = []
    for mcs in values:
        cm = run_hdbscan(density, min_cluster_size=mcs, min_samples=5)
        n = cm[cm["cluster_id"] >= 0]["cluster_id"].nunique()
        noise = int((cm["cluster_id"] == -1).sum())
        rows.append({"min_cluster_size": mcs, "n_clusters": int(n),
                     "noise": noise, "noise_pct": round(100 * noise / len(density), 2)})
        _log(f"  min_cluster_size={mcs:3d} -> {n:2d} clusters, noise={noise:5d} ({rows[-1]['noise_pct']}%)")
    return rows


def sweep_density_threshold(density: pd.DataFrame, min_cluster_size: int = 20,
                            percentiles=(0, 25, 50, 75, 90)) -> list[dict]:
    _log(f"=== 5.4b sweep: density threshold (fixed min_cluster_size={min_cluster_size}) ===")
    rows = []
    for p in percentiles:
        thr = int(density["count"].quantile(p / 100))
        sub = density[density["count"] >= thr]
        ms = max(5, min_cluster_size // 2)
        cm = run_hdbscan(sub, min_cluster_size=min_cluster_size, min_samples=ms)
        n = cm[cm["cluster_id"] >= 0]["cluster_id"].nunique()
        noise = int((cm["cluster_id"] == -1).sum())
        rows.append({"percentile": p, "threshold": thr, "n_input_cells": int(len(sub)),
                     "n_clusters": int(n), "noise": noise,
                     "noise_pct": round(100 * noise / len(sub), 2) if len(sub) else 0})
        _log(f"  top-{100-p:3d}% (>={thr:>6,}, n={len(sub):5d}) -> {n:2d} clusters, "
             f"noise={noise:4d} ({rows[-1]['noise_pct']}%)")
    return rows


def _plot_sensitivity(sweep_a: list[dict], sweep_b: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    xs = [r["min_cluster_size"] for r in sweep_a]
    ax = axes[0]
    ax.plot(xs, [r["n_clusters"] for r in sweep_a], "o-", color="steelblue", label="# clusters")
    ax.set_xlabel("min_cluster_size")
    ax.set_ylabel("Number of clusters", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["noise_pct"] for r in sweep_a], "s--", color="tomato", label="noise %")
    ax2.set_ylabel("Noise (%)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax.set_title("(A) Sensitivity to min_cluster_size\n(fixed input: all 33,284 cells)")

    xs2 = [100 - r["percentile"] for r in sweep_b]
    ax = axes[1]
    ax.plot(xs2, [r["n_clusters"] for r in sweep_b], "o-", color="steelblue", label="# clusters")
    ax.set_xlabel("Input = top X% densest cells")
    ax.set_ylabel("Number of clusters", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax.twinx()
    ax2.plot(xs2, [r["noise_pct"] for r in sweep_b], "s--", color="tomato", label="noise %")
    ax2.set_ylabel("Noise (%)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax.invert_xaxis()
    ax.set_title(f"(B) Sensitivity to density-input range\n(fixed min_cluster_size=20)")

    fig.suptitle("HDBSCAN Parameter Sensitivity – Systematic Sweeps", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_sensitivity_sweeps.png", dpi=150)
    plt.close()
    _log("  saved: hdbscan_sensitivity_sweeps.png")


# ── 5.6 temporal rhythm of top clusters ────────────────────────────────────────
def cluster_temporal_rhythm(df: pd.DataFrame, density: pd.DataFrame, top_n: int = 6) -> dict:
    _log(f"=== 5.6 temporal rhythm of top-{top_n} HDBSCAN (Run3) clusters ===")
    p90 = int(density["count"].quantile(0.90))
    hotspot = density[density["count"] >= p90].copy()
    cm_hot = run_hdbscan(hotspot, min_cluster_size=20, min_samples=10)

    cluster_visits = (cm_hot[cm_hot["cluster_id"] >= 0]
                      .groupby("cluster_id")["count"].sum()
                      .sort_values(ascending=False))
    top_clusters = cluster_visits.head(top_n).index.tolist()

    train = df[df["d"] <= 60]
    merged = train.merge(cm_hot[["x", "y", "cluster_id"]], on=["x", "y"], how="inner")

    results = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, cid in zip(axes.flat, top_clusters):
        sub = merged[merged["cluster_id"] == cid]
        wday = sub[sub["is_holiday"] == 0].groupby("t")["uid"].nunique().reindex(range(48), fill_value=0)
        hday = sub[sub["is_holiday"] == 1].groupby("t")["uid"].nunique().reindex(range(48), fill_value=0)
        peak_w = int(wday.idxmax())
        peak_h = int(hday.idxmax())
        ratio = float(wday.mean() / hday.mean()) if hday.mean() > 0 else float("nan")

        grp = cm_hot[cm_hot["cluster_id"] == cid]
        cx, cy = float(grp["x"].mean()), float(grp["y"].mean())
        lat = BBOX["south"] + cx * _LAT_STEP
        lon = BBOX["west"] + cy * _LON_STEP

        results.append({
            "cluster_id": int(cid),
            "n_cells": int(len(grp)),
            "total_visits": int(grp["count"].sum()),
            "center_lat": round(lat, 4), "center_lon": round(lon, 4),
            "weekday_peak": TIMES[peak_w], "holiday_peak": TIMES[peak_h],
            "weekday_holiday_ratio": round(ratio, 3),
            "weekday_mean_active": round(float(wday.mean()), 1),
            "holiday_mean_active": round(float(hday.mean()), 1),
        })

        ax.plot(range(48), wday.values, label="Weekday", color="steelblue", linewidth=1.5)
        ax.plot(range(48), hday.values, label="Holiday", color="tomato", linewidth=1.5)
        ax.set_xticks(range(0, 48, 8))
        ax.set_xticklabels([TIMES[t] for t in range(0, 48, 8)], fontsize=7)
        ax.set_title(f"Cluster {cid}  (cells={len(grp)}, visits={int(grp['count'].sum()):,})\n"
                     f"Wkday/Holiday={ratio:.2f}×  peak(wd)={TIMES[peak_w]}", fontsize=9)
        ax.set_xlabel("Time of Day", fontsize=7)
        ax.set_ylabel("Unique Users", fontsize=7)
        ax.legend(fontsize=7)

    for ax in axes.flat[len(top_clusters):]:
        ax.axis("off")

    fig.suptitle(f"Temporal Rhythm of Top-{top_n} HDBSCAN Clusters (Run3) – Weekday vs. Holiday", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hdbscan_cluster_temporal_rhythm.png", dpi=150, bbox_inches="tight")
    plt.close()
    _log("  saved: hdbscan_cluster_temporal_rhythm.png")
    for r in results:
        _log(f"  Cluster {r['cluster_id']:2d}: ratio={r['weekday_holiday_ratio']}x  "
             f"peak(wd)={r['weekday_peak']}  peak(hol)={r['holiday_peak']}")
    return {"top_n": top_n, "clusters": results}


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    train = df[df["d"] <= 60]
    density = compute_grid_density(train)
    _log(f"Density cells: {len(density):,}")

    out = {}
    out["core_distance"] = core_distance_contrast(density)
    out["comparison"] = run_comparison(density)
    sweep_a = sweep_min_cluster_size(density)
    sweep_b = sweep_density_threshold(density)
    _plot_sensitivity(sweep_a, sweep_b)
    out["sensitivity"] = {"sweep_min_cluster_size": sweep_a, "sweep_density_threshold": sweep_b}
    out["temporal_rhythm"] = cluster_temporal_rhythm(df, density, top_n=6)

    out_path = ROOT / "output" / "hdbscan_deepdive_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    _log(f"Saved {out_path}")


if __name__ == "__main__":
    main()
