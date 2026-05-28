import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

GRID_SIZE = 200

# 已知邊界框（由人流地圖視覺疊圖確認，2026-05-28）
BBOX = {
    "west":  136.50205100257344,
    "east":  137.49862964018885,
    "north": 35.50467287510108,
    "south": 34.49901520185085,
}

# x(0→199) = 南→北（lat 遞增）；y(0→199) = 西→東（lon 遞增）
_LAT_STEP = (BBOX["north"] - BBOX["south"]) / (GRID_SIZE - 1)
_LON_STEP = (BBOX["east"]  - BBOX["west"])  / (GRID_SIZE - 1)


# ── 時間處理 ──────────────────────────────────────────────────────────────────

def label_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use K-means (k=2) on daily active user counts to classify each day as
    working day (is_holiday=0) or holiday (is_holiday=1).
    The cluster with lower mean active users is labelled holiday.
    """
    daily_users = df.groupby("d")["uid"].nunique().reset_index()
    daily_users.columns = ["d", "active_users"]

    X = daily_users[["active_users"]].to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # cluster with lower centroid = holiday (fewer active users)
    centers = kmeans.cluster_centers_.ravel()
    holiday_cluster = int(np.argmin(centers))
    daily_users["is_holiday"] = (labels == holiday_cluster).astype("int8")

    df = df.merge(daily_users[["d", "is_holiday"]], on="d", how="left")
    return df


# ── 空間處理 ──────────────────────────────────────────────────────────────────

def grid_to_latlon(x: int, y: int) -> tuple[float, float]:
    """
    將單個網格座標 (x, y) 轉為中心點 (lat, lon)。
      x → 緯度（南→北）：lat = BBOX["south"] + x * _LAT_STEP
      y → 經度（西→東）：lon = BBOX["west"]  + y * _LON_STEP
    """
    lat = BBOX["south"] + x * _LAT_STEP
    lon = BBOX["west"]  + y * _LON_STEP
    return lat, lon


def build_grid_latlon_table(save_path: str = "data/grid_to_latlon.csv") -> pd.DataFrame:
    """
    建立全部 200×200 格子的座標對映表，存為 CSV。
    欄位：x, y, lat, lon, lat_min, lat_max, lon_min, lon_max
    """
    xs, ys = np.meshgrid(range(GRID_SIZE), range(GRID_SIZE), indexing="ij")
    xs = xs.ravel()
    ys = ys.ravel()

    lats = BBOX["south"] + xs * _LAT_STEP   # x=0 → south, x=199 → north
    lons = BBOX["west"]  + ys * _LON_STEP

    df = pd.DataFrame({
        "x": xs, "y": ys,
        "lat": lats, "lon": lons,
        "lat_min": lats - _LAT_STEP / 2,
        "lat_max": lats + _LAT_STEP / 2,
        "lon_min": lons - _LON_STEP / 2,
        "lon_max": lons + _LON_STEP / 2,
    })

    df.to_csv(save_path, index=False)
    print(f"[preprocessing] saved: {save_path}  ({len(df):,} rows)")
    return df


def build_grid_heatmap(df: pd.DataFrame) -> np.ndarray:
    """
    統計每個 (x, y) 格子的總出現次數（不含 x=999 佔位行）。
    回傳 shape (GRID_SIZE, GRID_SIZE)，density[y, x] = count。
    """
    real = df[df["x"] != 999]
    density = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    counts = real.groupby(["x", "y"]).size()
    for (x, y), cnt in counts.items():
        density[y, x] = cnt
    return density
