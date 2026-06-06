import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GRID_SIZE  = 200
OUTPUT_DIR = "output/figures"


def _orient(arr: np.ndarray) -> np.ndarray:
    """
    配合 origin='lower' 的方向轉換。
    density 儲存為 [y, x]（row=lon, col=lat），
    轉置後變成 [x, y]（row=lat, col=lon），
    搭配 origin='lower' 即可呈現正確地理方向（南↓北，西←東）。
    3D RGBA 陣列只轉置前兩軸。
    """
    if arr.ndim == 3:
        return np.transpose(arr, (1, 0, 2))
    return arr.T


def build_grid_heatmap(df: pd.DataFrame) -> np.ndarray:
    """
    統計每個 (x, y) 格子的總出現次數。
    回傳 shape (GRID_SIZE, GRID_SIZE)，density[y, x] = count。
    （顯示前需經 _orient 轉換）
    """
    density = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    counts = df.groupby(["x", "y"]).size()
    for (x, y), cnt in counts.items():
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            density[y, x] = cnt
    return density


def plot_spatial_heatmap(
    df: pd.DataFrame,
    save_path: str = f"{OUTPUT_DIR}/spatial_heatmap.png",
) -> None:
    """密度格子熱力圖（log scale，白底，可疊圖）。"""
    density = build_grid_heatmap(df)
    display = _orient(density)

    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(
        display,
        origin="lower",
        cmap="hot",
        norm=mcolors.LogNorm(
            vmin=max(1, density[density > 0].min()),
            vmax=density.max(),
        ),
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Visit Count (log scale)", fontsize=12
    )
    ax.set_title("Nagoya - Spatial Heatmap (all days)", fontsize=15)
    ax.set_xlabel("Grid Y (West -> East)")
    ax.set_ylabel("Grid X (South -> North)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eda] saved: {save_path}")


def plot_unique_users_map(
    df: pd.DataFrame,
    save_path: str = f"{OUTPUT_DIR}/unique_users_map.png",
) -> None:
    """
    每個 (x, y) 格子的唯一使用者數（仿 PDF 第 7 頁）。
    白底、半透明、可直接疊 OpenStreetMap。
    """
    unique = df.groupby(["x", "y"])["uid"].nunique()
    n_cells = len(unique)
    print(f"[eda] cells with data: {n_cells} / {GRID_SIZE * GRID_SIZE}")

    canvas = np.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=np.float32)
    vals = unique.values.astype(np.float32)
    log_max = np.log1p(vals).max()
    cmap = plt.get_cmap("Reds")

    for (x, y), cnt in unique.items():
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            continue
        norm_val = np.log1p(cnt) / log_max
        r, g, b, _ = cmap(norm_val)
        alpha = 0.25 + 0.75 * norm_val
        canvas[y, x] = [r, g, b, alpha]

    display = _orient(canvas)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")
    ax.imshow(display, origin="lower", interpolation="nearest")
    ax.set_title(
        f"Nagoya - Unique Users per Cell (all {df['d'].nunique()} days)\n"
        f"{n_cells} cells with data",
        fontsize=13,
    )
    ax.set_xlabel("Grid Y (West -> East)")
    ax.set_ylabel("Grid X (South -> North)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="white")
    plt.close()
    print(f"[eda] saved: {save_path}")


def build_trajectory_density(
    df: pd.DataFrame,
    n_users: int = 10000,
    time_slice: tuple[int, int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Bresenham 線段累積軌跡密度，回傳 density[y, x]。"""
    rng = np.random.default_rng(seed)
    uids = df["uid"].unique()
    sampled = rng.choice(uids, size=min(n_users, len(uids)), replace=False)
    sub = df[df["uid"].isin(sampled)]

    if time_slice is not None:
        sub = sub[sub["t"].between(time_slice[0], time_slice[1])]

    sub = sub.sort_values(["uid", "d", "t"])
    density = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def bresenham(x0, y0, x1, y1):
        pts = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            pts.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    for (_, __), g in sub.groupby(["uid", "d"], sort=False):
        if len(g) < 2:
            continue
        xs = g["x"].to_numpy(dtype=np.int16)
        ys = g["y"].to_numpy(dtype=np.int16)
        for i in range(len(xs) - 1):
            for cx, cy in bresenham(int(xs[i]), int(ys[i]), int(xs[i+1]), int(ys[i+1])):
                if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                    density[cy, cx] += 1

    return density


def plot_trajectory_map(
    df: pd.DataFrame,
    n_users: int = 10000,
    time_slice: tuple[int, int] | None = None,
    save_path: str = f"{OUTPUT_DIR}/trajectory_map.png",
    seed: int = 42,
) -> None:
    """軌跡密度地圖（log scale，黑底）。"""
    density = build_trajectory_density(df, n_users=n_users, time_slice=time_slice, seed=seed)
    display = _orient(np.log1p(density))

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(display, origin="lower", cmap="hot", vmin=0, vmax=display.max(), interpolation="bilinear")

    t_label = f"t={time_slice[0]}-{time_slice[1]}" if time_slice else "all day"
    ax.set_title(
        f"Nagoya - Trajectory Density ({t_label}), {n_users} users, log scale",
        color="white", fontsize=13,
    )
    ax.set_xlabel("Grid Y (West -> East)", color="white")
    ax.set_ylabel("Grid X (South -> North)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor="black")
    plt.close()
    print(f"[eda] saved: {save_path}")


def plot_daily_active_users(
    df: pd.DataFrame,
    save_path: str = f"{OUTPUT_DIR}/daily_active_users.png",
) -> None:
    """折線圖：每日活躍 uid 數，d=60 為訓練/測試切分線。"""
    daily = df.groupby("d")["uid"].nunique().reset_index()
    daily.columns = ["d", "active_users"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily["d"], daily["active_users"], linewidth=1, color="steelblue")
    ax.axvline(x=60, color="red", linestyle="--", linewidth=1.2, label="Train/Test split (d=60)")
    ax.set_title("Nagoya - Daily Active Users", fontsize=14)
    ax.set_xlabel("Day")
    ax.set_ylabel("Active Users")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eda] saved: {save_path}")
