from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis.clustering import (
    assign_user_hotspots,
    assign_poi_to_grid,
    build_grid_poi_features,
    compute_grid_density,
    fetch_poi_from_osm,
    run_hdbscan,
)
from analysis.trajectory import build_user_stability_features
from data.loader import load_city, split_train_test
from data.preprocessing import build_grid_latlon_table, label_holidays
from models.baseline import align_prediction_to_reference, run_all_baselines
from models.cvae import CVAE, build_condition_table, predict_trajectories, train_cvae
from eval.metrics import compute_fde, compute_geobleu, generate_report


def run_preprocessing(
    city_path: str,
    max_users: int | None = None,
    sample_users: int | None = None,
    random_seed: int = 42,
    chunksize: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_city(
        city_path,
        max_users=max_users,
        sample_users=sample_users,
        random_seed=random_seed,
        chunksize=chunksize,
    )
    df = label_holidays(df)
    train_df, test_df = split_train_test(df)
    grid_path = Path("data/grid_to_latlon.csv")
    if grid_path.exists():
        grid_latlon = pd.read_csv(grid_path)
    else:
        grid_latlon = build_grid_latlon_table(str(grid_path))
    return train_df, test_df, grid_latlon


def run_feature_engineering(
    train_df: pd.DataFrame,
    grid_latlon: pd.DataFrame,
    fetch_poi: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    density = compute_grid_density(train_df)
    cluster_map = run_hdbscan(density)
    user_hotspots = assign_user_hotspots(train_df, cluster_map)
    user_stability = build_user_stability_features(train_df)
    if fetch_poi:
        poi_df = fetch_poi_from_osm(grid_latlon)
        poi_df.to_csv("data/osm_poi_raw.csv", index=False)
        poi_grid = assign_poi_to_grid(poi_df, grid_latlon)
        poi_grid.to_csv("data/osm_poi_grid.csv", index=False)
    else:
        poi_grid = pd.DataFrame(columns=["x", "y", "poi_type", "osm_id"])
    grid_poi_features = build_grid_poi_features(poi_grid)
    user_hotspots.to_csv("data/user_hotspots.csv", index=False)
    cluster_map.to_csv("data/grid_clusters.csv", index=False)
    return cluster_map, user_hotspots, user_stability, grid_poi_features


def run_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    scores = run_all_baselines(train_df, test_df)
    for name, score in scores.items():
        print(f"{name}: GEO-BLEU={score:.6f}")


def run_cvae(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_map: pd.DataFrame,
    user_hotspots: pd.DataFrame,
    user_stability: pd.DataFrame,
    grid_poi: pd.DataFrame,
    epochs: int = 1,
    batch_size: int = 256,
) -> None:
    import json, shutil
    n_users = train_df["uid"].nunique()
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"cvae_ep{epochs}_u{n_users}.pt"
    best_path = ckpt_dir / "cvae_best.pt"
    best_meta = ckpt_dir / "cvae_best_meta.json"

    train_days = sorted(train_df["d"].unique().astype(int).tolist())
    test_days = sorted(test_df["d"].unique().astype(int).tolist())
    uids = sorted(train_df["uid"].unique().astype(int).tolist())
    condition_df = build_condition_table(
        uids=uids,
        days=sorted(set(train_days + test_days)),
        user_hotspots=user_hotspots,
        user_stability=user_stability,
        grid_poi=grid_poi,
        cluster_map=cluster_map,
    )
    condition_dim = int(len(condition_df.iloc[0]["condition"]))
    model = CVAE(condition_dim=condition_dim)
    train_conditions = condition_df[condition_df["d"].isin(train_days)].reset_index(drop=True)
    model = train_cvae(
        train_df=train_df,
        condition_df=train_conditions,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=str(ckpt_path),
    )
    print(f"[cvae] checkpoint saved → {ckpt_path}")

    pred = predict_trajectories(model, condition_df, test_days)
    pred = align_prediction_to_reference(pred, test_df)
    pred.to_csv("eval/reports/cvae_predictions.csv", index=False)
    geobleu_result = compute_geobleu(pred, test_df)
    fde_result = compute_fde(pred, test_df)
    generate_report("cvae", geobleu_result, fde_result)
    score = float(geobleu_result.get("mean", 0.0))
    print(f"[cvae] GEO-BLEU={score:.6f}  (epochs={epochs}, users={n_users})")

    best_score = 0.0
    if best_meta.exists():
        best_score = json.loads(best_meta.read_text()).get("geobleu", 0.0)
    if score > best_score:
        shutil.copy2(ckpt_path, best_path)
        best_meta.write_text(json.dumps({"geobleu": score, "epochs": epochs, "users": n_users}, indent=2))
        print(f"[cvae] ★ new best! GEO-BLEU={score:.6f} → {best_path}")
    else:
        print(f"[cvae] best unchanged (best={best_score:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="DataMining final project pipeline")
    parser.add_argument("--city-path", type=str, help="Path to city CSV, e.g. raw_data/nagoya_challengedata.csv")
    parser.add_argument("--max-users", type=int, help="Use the first N uid values after sorting")
    parser.add_argument("--sample-users", type=int, help="Randomly sample N uid values")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, help="Read CSV in chunks of this many rows")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--fetch-poi", action="store_true", help="Fetch POI data from OpenStreetMap Overpass API")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--run-cvae", action="store_true")
    parser.add_argument("--cvae-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not args.city_path:
        print("Provide --city-path to run the pipeline.")
        return
    if not Path(args.city_path).exists():
        raise FileNotFoundError(args.city_path)

    train_df, test_df, grid_latlon = run_preprocessing(
        args.city_path,
        max_users=args.max_users,
        sample_users=args.sample_users,
        random_seed=args.random_seed,
        chunksize=args.chunksize,
    )
    print(f"train rows={len(train_df):,}, test rows={len(test_df):,}")

    feature_outputs = None
    if not args.skip_features or args.run_cvae:
        feature_outputs = run_feature_engineering(train_df, grid_latlon, fetch_poi=args.fetch_poi)
    if args.run_baselines:
        run_baselines(train_df, test_df)
    if args.run_cvae:
        if feature_outputs is None:
            raise RuntimeError("CVAE requires feature engineering outputs")
        cluster_map, user_hotspots, user_stability, grid_poi = feature_outputs
        run_cvae(
            train_df,
            test_df,
            cluster_map,
            user_hotspots,
            user_stability,
            grid_poi,
            epochs=args.cvae_epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
