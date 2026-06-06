"""
Personal trajectory analysis - four methods inspired by classmate's approach:
1. Per-user location standard deviation (working vs non-working)
2. DTW commute regularity (t=16~36, working days)
3. User feature extraction (home/work/hotspot) -> city-level POI inference
4. Activity space HDBSCAN clustering (life zone grouping by bbox)
"""
from __future__ import annotations

import io
import json
import sys
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

HOLIDAY_DAYS = {0,1,2,3,4,5,6,7,8,13,14,20,21,27,28,35,37,41,42,45,46,47,48,49,50,56}
TIMES = [f"{(t*30)//60:02d}:{(t*30)%60:02d}" for t in range(48)]

# Known landmarks for annotation
KNOWN_POIS = {
    "名古屋車站": (133, 76),
    "名古屋大學": (131, 87),
    "名古屋城": (136, 80),
    "大須商店街": (131, 80),
    "金山": (129, 81),
}


def _log(msg: str) -> None:
    print(f"[personal] {msg}", flush=True)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_train_data() -> pd.DataFrame:
    _log("Loading parquet (training days d≤60) …")
    df = pd.read_parquet(ROOT / "output" / "nagoya_clean.parquet")
    df = df[df["d"] <= 60].copy()
    df["is_holiday"] = df["d"].isin(HOLIDAY_DAYS).astype("int8")
    _log(f"  {len(df):,} rows  {df['uid'].nunique():,} users  {df['d'].nunique()} days")
    return df


# ── 1. Per-user std stability ─────────────────────────────────────────────────

def compute_user_std(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (uid, t), compute std of x and y across days.
    Then average across time slots per user.
    Returns one row per (uid, day_type) with x_std_mean, y_std_mean, std_mean.
    """
    _log("Computing per-user std (working / non-working) …")
    results = []
    for day_flag, label in [(0, "working"), (1, "non_working")]:
        sub = df[df["is_holiday"] == day_flag]
        # std across days for each (uid, t)
        std_t = (
            sub.groupby(["uid", "t"])[["x", "y"]]
            .std()
            .dropna()          # drop t-slots with only 1 observation
            .reset_index()
        )
        user_std = std_t.groupby("uid")[["x", "y"]].mean().reset_index()
        user_std.columns = ["uid", "x_std_mean", "y_std_mean"]
        user_std["std_mean"] = (user_std["x_std_mean"] + user_std["y_std_mean"]) / 2
        user_std["day_type"] = label
        results.append(user_std)
        _log(f"  {label}: {len(user_std):,} users")
    return pd.concat(results, ignore_index=True)


def plot_std_distribution(user_std_df: pd.DataFrame) -> dict:
    _log("Plotting std distributions …")
    bins   = [0, 1, 2, 3, 5, 10, 20, 999]
    labels = ["<1", "1-2", "2-3", "3-5", "5-10", "10-20", "≥20"]

    stats: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (day_type, color, title) in zip(
        axes,
        [("working", "steelblue", "工作日"), ("non_working", "tomato", "假日/週末")],
    ):
        sub = user_std_df[user_std_df["day_type"] == day_type].copy()
        sub["std_bin"] = pd.cut(sub["std_mean"], bins=bins, labels=labels, right=False)
        dist = sub["std_bin"].value_counts().reindex(labels, fill_value=0)
        total = len(sub)

        ax.bar(dist.index.astype(str), dist.values, color=color, alpha=0.85, edgecolor="white")
        for i, (lbl, cnt) in enumerate(dist.items()):
            ax.text(i, cnt + 100, f"{100*cnt/total:.1f}%", ha="center", fontsize=8)
        ax.set_title(f"Nagoya {title} - 使用者位置標準差分布")
        ax.set_xlabel("平均 std（格距，1格≈500m）")
        ax.set_ylabel("使用者數")

        stats[day_type] = {
            "distribution": {
                lbl: {"count": int(cnt), "pct": round(100 * cnt / total, 1)}
                for lbl, cnt in dist.items()
            },
            "mean_std": round(float(sub["std_mean"].mean()), 2),
            "median_std": round(float(sub["std_mean"].median()), 2),
            "very_stable_pct": round(100 * (sub["std_mean"] < 5).sum() / total, 1),
        }

    plt.tight_layout()
    path = FIG_DIR / "personal_std_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved → {path.name}")
    return stats


# ── 2. DTW commute regularity ─────────────────────────────────────────────────

def compute_dtw_regularity(
    df: pd.DataFrame,
    user_std_df: pd.DataFrame,
    n_sample: int = 3000,
) -> dict:
    """
    For users with 5 ≤ working-day std_mean ≤ 20, compute per-user DTW
    distance between each working-day commute trajectory (t=16~36) and the
    mean trajectory across all training working days.
    """
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    _log("Running DTW commute regularity …")

    working_std = user_std_df[user_std_df["day_type"] == "working"]
    eligible = working_std[working_std["std_mean"].between(5, 20)]
    _log(f"  eligible users (5≤std≤20): {len(eligible):,}")

    sampled_uids = set(
        eligible["uid"].sample(min(n_sample, len(eligible)), random_state=42).tolist()
    )

    # Filter data: working days, commute time slots
    T_SLOTS = list(range(16, 37))   # 08:00–18:00 (21 slots)
    commute = (
        df[(df["is_holiday"] == 0) & (df["uid"].isin(sampled_uids)) & df["t"].between(16, 36)]
        .groupby(["uid", "d", "t"])[["x", "y"]]
        .mean()
        .reset_index()
    )

    # Re-use cached results if already computed
    cache_path = ROOT / "output" / "dtw_cache.parquet"
    if cache_path.exists():
        _log("  loading cached DTW results …")
        dtw_df = pd.read_parquet(cache_path)
        _log(f"  cached DTW for {len(dtw_df):,} users")
        return _dtw_stats(dtw_df, len(eligible))

    dtw_rows = []
    uids = commute["uid"].unique()
    _log(f"  computing DTW for {len(uids):,} sampled users …")

    for idx, uid in enumerate(uids):
        if idx % 500 == 0:
            _log(f"    {idx}/{len(uids)} …")

        user_data = commute[commute["uid"] == uid]
        days = user_data["d"].unique()
        if len(days) < 3:
            continue

        trajs = []
        for d in days:
            day_traj = (
                user_data[user_data["d"] == d]
                .set_index("t")[["x", "y"]]
                .reindex(T_SLOTS)
                .ffill()
                .bfill()
            )
            if day_traj.isna().any().any():
                continue
            trajs.append(day_traj.to_numpy(dtype=float))

        if len(trajs) < 3:
            continue

        mean_traj = np.mean(trajs, axis=0)
        dists = [fastdtw(t, mean_traj, dist=euclidean)[0] for t in trajs]

        row = working_std[working_std["uid"] == uid].iloc[0]
        dtw_rows.append({
            "uid": uid,
            "mean_dtw": float(np.mean(dists)),
            "n_days": len(trajs),
            "x_std": float(row["x_std_mean"]),
            "y_std": float(row["y_std_mean"]),
        })

    dtw_df = pd.DataFrame(dtw_rows)
    dtw_df.to_parquet(cache_path, index=False)
    _log(f"  DTW computed for {len(dtw_df):,} users  (cached to dtw_cache.parquet)")

    return _dtw_stats(dtw_df, len(eligible))


def _dtw_stats(dtw_df: pd.DataFrame, n_eligible: int) -> dict:
    # DTW scale: 21 time slots × euclidean(x,y); thresholds reflect grid distance
    # < 100  → avg ~5 grid cells / slot  → very regular commute
    # 100-300 → avg 5-14 grid cells / slot → moderate variation
    # > 300  → avg > 14 grid cells / slot  → irregular movement
    very_regular = int((dtw_df["mean_dtw"] < 100).sum())
    regular      = int(((dtw_df["mean_dtw"] >= 100) & (dtw_df["mean_dtw"] < 300)).sum())
    irregular    = int((dtw_df["mean_dtw"] >= 300).sum())
    corr = float(
        (dtw_df["x_std"] + dtw_df["y_std"]).corr(dtw_df["mean_dtw"])
    )

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(dtw_df["mean_dtw"], bins=50, color="steelblue", alpha=0.85, edgecolor="white")
    med = dtw_df["mean_dtw"].median()
    ax.axvline(med, color="red", linestyle="--", label=f"Median: {med:.1f}")
    ax.set_title("DTW Mean Distance Distribution (Commute Hours 08:00-18:00)")
    ax.set_xlabel("Mean DTW Distance")
    ax.set_ylabel("User Count")
    ax.legend()

    ax = axes[1]
    ax.scatter(
        dtw_df["x_std"] + dtw_df["y_std"],
        dtw_df["mean_dtw"],
        alpha=0.25, s=5, color="steelblue",
    )
    ax.set_title(f"Location Std vs DTW  (r={corr:.2f})")
    ax.set_xlabel("x_std + y_std (movement range)")
    ax.set_ylabel("Mean DTW Distance (route regularity)")

    plt.tight_layout()
    path = FIG_DIR / "personal_dtw_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved → {path.name}")

    return {
        "n_eligible": n_eligible,
        "n_analyzed": int(len(dtw_df)),
        "mean_dtw": round(float(dtw_df["mean_dtw"].mean()), 2),
        "median_dtw": round(float(dtw_df["mean_dtw"].median()), 2),
        "very_regular_users_pct": round(100 * very_regular / len(dtw_df), 1),
        "regular_users_pct": round(100 * regular / len(dtw_df), 1),
        "irregular_users_pct": round(100 * irregular / len(dtw_df), 1),
        "correlation_std_dtw": round(corr, 3),
    }


# ── 3. User feature extraction → POI inference ───────────────────────────────

def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised extraction of home, work, top-hotspot, and bounding box per user.
      home    : working day, t≤12 or t≥40, most frequent (x,y)
      work    : working day, t=16~34, most frequent (x,y)
      hotspot : all training days, overall most frequent (x,y)
    """
    _log("Extracting user features (vectorised) …")
    working = df[df["is_holiday"] == 0]

    def _mode_xy(sub: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        return (
            sub.groupby(["uid", "x", "y"]).size()
            .reset_index(name="_cnt")
            .sort_values("_cnt", ascending=False)
            .drop_duplicates("uid")
            .rename(columns={"x": x_col, "y": y_col})[["uid", x_col, y_col]]
        )

    home_data    = working[(working["t"] <= 12) | (working["t"] >= 40)]
    work_data    = working[working["t"].between(16, 34)]
    hotspot_data = df  # all training days

    home_df    = _mode_xy(home_data,    "home_x",    "home_y")
    work_df    = _mode_xy(work_data,    "work_x",    "work_y")
    hotspot_df = _mode_xy(hotspot_data, "hotspot_x", "hotspot_y")

    bbox = df.groupby("uid").agg(
        bbox_xmin=("x", "min"),
        bbox_ymin=("y", "min"),
        bbox_xmax=("x", "max"),
        bbox_ymax=("y", "max"),
    ).reset_index()

    features = (
        bbox
        .merge(home_df,    on="uid", how="left")
        .merge(work_df,    on="uid", how="left")
        .merge(hotspot_df, on="uid", how="left")
    )
    _log(f"  features for {len(features):,} users")
    return features


def extract_poi_from_features(
    features_df: pd.DataFrame,
    top_n: int = 30,
    min_dist: int = 2,
) -> pd.DataFrame:
    """
    Aggregate home / work / hotspot points across users; keep top_n POIs
    filtered by minimum grid distance.
    """
    _log("Aggregating city-level POIs from user features …")
    pts = []
    for prefix in ["home", "work", "hotspot"]:
        xc, yc = f"{prefix}_x", f"{prefix}_y"
        if xc in features_df.columns:
            sub = features_df[[xc, yc]].dropna().astype(int)
            sub.columns = ["x", "y"]
            pts.append(sub)

    all_pts = pd.concat(pts, ignore_index=True)
    freq = (
        all_pts.groupby(["x", "y"]).size()
        .sort_values(ascending=False)
        .reset_index(name="freq")
    )

    selected: list[tuple[int, int]] = []
    for _, row in freq.iterrows():
        x, y = int(row["x"]), int(row["y"])
        if all(np.hypot(x - sx, y - sy) >= min_dist for sx, sy in selected):
            selected.append((x, y))
        if len(selected) >= top_n:
            break

    poi_df = pd.DataFrame(selected, columns=["x", "y"])
    poi_df["lat"] = BBOX["south"] + poi_df["x"] * _LAT_STEP
    poi_df["lon"] = BBOX["west"]  + poi_df["y"] * _LON_STEP
    freq_lookup = freq.set_index(["x", "y"])["freq"]
    poi_df["freq"] = poi_df.apply(
        lambda r: int(freq_lookup.get((r["x"], r["y"]), 0)), axis=1
    )
    _log(f"  {len(poi_df)} POIs extracted")
    return poi_df


def plot_user_poi_map(poi_df: pd.DataFrame, df: pd.DataFrame) -> None:
    _log("Plotting user-derived POI map …")

    grid_visits = df.groupby(["x", "y"]).size().reset_index(name="visits")
    canvas = np.zeros((GRID_SIZE, GRID_SIZE))
    for _, r in grid_visits.iterrows():
        xi, yi = int(r["x"]), int(r["y"])
        if 0 <= xi < GRID_SIZE and 0 <= yi < GRID_SIZE:
            canvas[xi, yi] = r["visits"]

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(
        np.log1p(canvas).T, origin="lower", cmap="YlOrRd",
        aspect="equal", interpolation="bilinear",
    )
    plt.colorbar(im, ax=ax, label="log(visits)", shrink=0.7)

    # User-derived POIs
    sizes = poi_df["freq"].clip(upper=poi_df["freq"].quantile(0.95))
    sizes = 30 + 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1)
    ax.scatter(
        poi_df["x"], poi_df["y"],
        s=sizes, c=poi_df["freq"], cmap="Blues_r",
        zorder=5, edgecolors="navy", linewidths=0.6, alpha=0.85,
        label="行為反推 POI",
    )

    # Known landmarks
    for name, (x, y) in KNOWN_POIS.items():
        ax.plot(x, y, "r*", markersize=12, zorder=6)
        ax.annotate(
            name, (x, y), textcoords="offset points", xytext=(5, 5),
            fontsize=7.5, color="red", fontweight="bold",
        )

    ax.set_title("個人行為特徵反推 POI（深藍圈）vs 已知地標（紅星）", fontsize=13)
    ax.set_xlabel("Grid X（South → North）")
    ax.set_ylabel("Grid Y（West → East）")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = FIG_DIR / "personal_poi_map.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved → {path.name}")


# ── 4. Activity space HDBSCAN clustering ─────────────────────────────────────

def activity_space_cluster(features_df: pd.DataFrame) -> dict:
    """
    HDBSCAN on user bounding-box vectors to find life-zone groups.
    """
    import hdbscan as _hdbscan

    _log("Running activity space HDBSCAN clustering …")
    bbox_cols = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    feat = features_df[["uid"] + bbox_cols].dropna().copy()
    X = feat[bbox_cols].to_numpy(dtype=float)

    clusterer = _hdbscan.HDBSCAN(min_cluster_size=300, min_samples=350)
    labels = clusterer.fit_predict(X)
    feat = feat.copy()
    feat["cluster"] = labels

    unique_labels = sorted(set(labels))
    n_clusters = sum(1 for c in unique_labels if c >= 0)
    n_noise     = int((labels == -1).sum())
    total       = len(labels)
    _log(f"  clusters={n_clusters}  noise={n_noise} ({100*n_noise/total:.1f}%)")

    # Cluster summary
    summary = []
    for cid in unique_labels:
        sub = feat[feat["cluster"] == cid]
        cx = float((sub["bbox_xmin"] + sub["bbox_xmax"]).mean() / 2)
        cy = float((sub["bbox_ymin"] + sub["bbox_ymax"]).mean() / 2)
        wx = float((sub["bbox_xmax"] - sub["bbox_xmin"]).mean())
        wy = float((sub["bbox_ymax"] - sub["bbox_ymin"]).mean())
        summary.append({
            "cluster_id": int(cid),
            "n_users": int(len(sub)),
            "pct": round(100 * len(sub) / total, 1),
            "center_x": round(cx, 1),
            "center_y": round(cy, 1),
            "mean_range_x": round(wx, 1),
            "mean_range_y": round(wy, 1),
        })

    # ── figures ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart
    ax = axes[0]
    cid_labels = [f"C{r['cluster_id']}" if r["cluster_id"] >= 0 else "noise" for r in summary]
    bar_colors = [
        "lightgray" if r["cluster_id"] == -1 else f"C{r['cluster_id']}"
        for r in summary
    ]
    ax.bar(cid_labels, [r["n_users"] for r in summary], color=bar_colors, alpha=0.85, edgecolor="white")
    for i, r in enumerate(summary):
        ax.text(i, r["n_users"] + 100, f"{r['pct']}%", ha="center", fontsize=8)
    ax.set_title("生活圈聚類 — 各群使用者數")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("使用者數")

    # Scatter: bbox center coloured by cluster
    ax2 = axes[1]
    feat["cx"] = (feat["bbox_xmin"] + feat["bbox_xmax"]) / 2
    feat["cy"] = (feat["bbox_ymin"] + feat["bbox_ymax"]) / 2
    for cid in unique_labels:
        sub = feat[feat["cluster"] == cid]
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=42)
        color = "lightgray" if cid == -1 else f"C{cid}"
        lbl   = "noise" if cid == -1 else f"Cluster {cid}"
        alpha = 0.08 if cid == -1 else 0.5
        ax2.scatter(sub["cx"], sub["cy"], s=2, alpha=alpha, color=color, label=lbl)

    # Overlay known POIs
    for name, (x, y) in KNOWN_POIS.items():
        ax2.plot(x, y, "r*", markersize=10, zorder=6)
        ax2.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3),
                     fontsize=7, color="red")

    ax2.set_xlim(0, GRID_SIZE)
    ax2.set_ylim(0, GRID_SIZE)
    ax2.set_title("生活圈聚類 — 使用者活動中心分布")
    ax2.set_xlabel("Grid X（South → North）")
    ax2.set_ylabel("Grid Y（West → East）")
    ax2.legend(markerscale=5, fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "activity_space_clusters.png"
    plt.savefig(path, dpi=150)
    plt.close()
    _log(f"  saved → {path.name}")

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(100 * n_noise / total, 1),
        "cluster_summary": summary,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_train_data()
    all_stats: dict = {}

    # 1. Per-user std
    user_std_df = compute_user_std(df)
    std_stats = plot_std_distribution(user_std_df)
    all_stats["user_std"] = std_stats
    user_std_df.to_parquet(ROOT / "output" / "user_stability_std.parquet", index=False)
    _log("Saved user_stability_std.parquet")

    # 2. DTW
    dtw_stats = compute_dtw_regularity(df, user_std_df, n_sample=3000)
    all_stats["dtw"] = dtw_stats
    _log(f"DTW summary: {dtw_stats}")

    # 3. Feature extraction + POI
    features_df = extract_user_features(df)
    poi_df = extract_poi_from_features(features_df, top_n=30, min_dist=2)
    plot_user_poi_map(poi_df, df)
    features_df.to_parquet(ROOT / "output" / "user_features.parquet", index=False)
    poi_df.to_json(
        ROOT / "output" / "user_derived_pois.json",
        orient="records", indent=2, force_ascii=False,
    )
    all_stats["user_derived_poi"] = {
        "n_pois": int(len(poi_df)),
        "top10": poi_df.head(10).to_dict("records"),
    }
    _log("Saved user_features.parquet and user_derived_pois.json")

    # 4. Activity space clustering
    cluster_stats = activity_space_cluster(features_df)
    all_stats["activity_space_clusters"] = cluster_stats

    # Save combined stats
    out_path = ROOT / "output" / "personal_trajectory_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    _log(f"Saved personal_trajectory_stats.json")
    _log("Done!")


if __name__ == "__main__":
    main()
