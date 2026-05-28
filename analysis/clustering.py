from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GRID_SIZE = 200
DEFAULT_POI_COLUMNS = [
    "poi_station",
    "poi_school",
    "poi_hospital",
    "poi_restaurant",
    "poi_shop",
    "poi_leisure",
]

POI_TYPE_MAP = {
    "station": "station",
    "school": "school",
    "hospital": "hospital",
    "restaurant": "restaurant",
    "shop": "shop",
    "leisure": "leisure",
}


def compute_grid_density(df: pd.DataFrame) -> pd.DataFrame:
    """Count visits per grid cell."""
    return df.groupby(["x", "y"]).size().rename("count").reset_index()


def run_hdbscan(
    density_df: pd.DataFrame,
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Cluster occupied grid cells with HDBSCAN."""
    out = density_df.copy()
    if out.empty:
        out["cluster_id"] = pd.Series(dtype="int64")
        return out
    coords = out[["x", "y"]].to_numpy(dtype=float)
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(coords)
    except ImportError:
        from sklearn.cluster import DBSCAN

        eps = max(1.5, float(min_cluster_size) ** 0.5)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    out["cluster_id"] = labels.astype(int)
    return out


def assign_user_hotspots(
    df: pd.DataFrame,
    cluster_map: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Assign each user their top-N visited hotspot clusters."""
    merged = df.merge(cluster_map[["x", "y", "cluster_id"]], on=["x", "y"], how="left")
    merged["cluster_id"] = merged["cluster_id"].fillna(-1).astype(int)
    rows = []
    for uid, group in merged.groupby("uid", sort=False):
        counts = group[group["cluster_id"].ne(-1)]["cluster_id"].value_counts()
        hotspots = counts.index.astype(int).tolist()[:top_n]
        hotspots += [-1] * (top_n - len(hotspots))
        rows.append({"uid": int(uid), **{f"hotspot_{idx}": value for idx, value in enumerate(hotspots)}})
    return pd.DataFrame(rows)


def fetch_poi_from_osm(
    grid_latlon: pd.DataFrame,
    poi_tags: dict[str, str | list[str] | bool] | None = None,
) -> pd.DataFrame:
    """Fetch POIs in the grid bounding box using Overpass."""
    import overpy

    if poi_tags is None:
        poi_tags = {
            "amenity": ["station", "school", "hospital", "restaurant"],
            "shop": True,
            "leisure": True,
        }
    south = float(grid_latlon["lat_min"].min())
    west = float(grid_latlon["lon_min"].min())
    north = float(grid_latlon["lat_max"].max())
    east = float(grid_latlon["lon_max"].max())

    filters = []
    for key, value in poi_tags.items():
        if value is True:
            filters.append(f'node["{key}"]({south},{west},{north},{east});')
        elif isinstance(value, list):
            pattern = "|".join(map(str, value))
            filters.append(f'node["{key}"~"^({pattern})$"]({south},{west},{north},{east});')
        else:
            filters.append(f'node["{key}"="{value}"]({south},{west},{north},{east});')
    query = "[out:json][timeout:120];(" + "".join(filters) + ");out body;"
    result = overpy.Overpass().query(query)

    rows = []
    for node in result.nodes:
        tags = dict(node.tags)
        raw_type = (
            tags.get("amenity")
            or ("shop" if "shop" in tags else None)
            or ("leisure" if "leisure" in tags else None)
            or "unknown"
        )
        poi_type = POI_TYPE_MAP.get(raw_type, raw_type)
        rows.append({"osm_id": int(node.id), "poi_type": poi_type, "lat": float(node.lat), "lon": float(node.lon)})
    return pd.DataFrame(rows, columns=["osm_id", "poi_type", "lat", "lon"])


def assign_poi_to_grid(poi_df: pd.DataFrame, grid_latlon: pd.DataFrame) -> pd.DataFrame:
    """Assign each POI to the nearest grid center using the known affine grid."""
    if poi_df.empty:
        return pd.DataFrame(columns=["x", "y", "poi_type", "osm_id"])
    lat_step = (grid_latlon["lat"].max() - grid_latlon["lat"].min()) / (GRID_SIZE - 1)
    lon_step = (grid_latlon["lon"].max() - grid_latlon["lon"].min()) / (GRID_SIZE - 1)
    lat0 = grid_latlon["lat"].min()
    lon0 = grid_latlon["lon"].min()
    out = poi_df.copy()
    out["x"] = np.rint((out["lat"] - lat0) / lat_step).clip(0, GRID_SIZE - 1).astype(int)
    out["y"] = np.rint((out["lon"] - lon0) / lon_step).clip(0, GRID_SIZE - 1).astype(int)
    return out[["x", "y", "poi_type", "osm_id"]]


def build_grid_poi_features(poi_grid: pd.DataFrame) -> pd.DataFrame:
    """Build per-grid POI count features and save them."""
    if poi_grid.empty:
        xs, ys = np.meshgrid(range(GRID_SIZE), range(GRID_SIZE), indexing="ij")
        features = pd.DataFrame({"x": xs.ravel(), "y": ys.ravel()})
        for column in DEFAULT_POI_COLUMNS:
            features[column] = 0
        features["poi_total"] = 0
    else:
        normalized = poi_grid.copy()
        normalized["poi_type"] = normalized["poi_type"].map(lambda value: POI_TYPE_MAP.get(str(value), str(value)))
        pivot = (
            normalized.assign(column="poi_" + normalized["poi_type"].astype(str))
            .pivot_table(index=["x", "y"], columns="column", values="osm_id", aggfunc="count", fill_value=0)
            .reset_index()
        )
        all_cells = pd.DataFrame(
            {
                "x": np.repeat(np.arange(GRID_SIZE), GRID_SIZE),
                "y": np.tile(np.arange(GRID_SIZE), GRID_SIZE),
            }
        )
        pivot = all_cells.merge(pivot, on=["x", "y"], how="left").fillna(0)
        for column in DEFAULT_POI_COLUMNS:
            if column not in pivot:
                pivot[column] = 0
        poi_cols = [column for column in pivot.columns if column.startswith("poi_")]
        pivot["poi_total"] = pivot[poi_cols].sum(axis=1)
        features = pivot[["x", "y", *DEFAULT_POI_COLUMNS, "poi_total"]]
    path = Path("data/grid_poi_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return features
