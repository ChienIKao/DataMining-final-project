"""
Master script: create parquet cache + run all analysis scripts.
Outputs to output/figures/, then copies to docs/figures/ and report/img/.
Usage: uv run --frozen python scripts/run_all_analysis.py
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_FIG = ROOT / "output" / "figures"
DOCS_FIG   = ROOT / "docs" / "figures"
REPORT_IMG = ROOT / "report" / "img"

for d in [OUTPUT_FIG, DOCS_FIG, REPORT_IMG]:
    d.mkdir(parents=True, exist_ok=True)


def step0_create_parquet():
    parquet_path = ROOT / "output" / "nagoya_clean.parquet"
    if parquet_path.exists():
        print("[runner] parquet already exists, skipping creation")
        return
    print("[runner] creating nagoya_clean.parquet ...")
    from data.loader import load_city
    from data.preprocessing import label_holidays
    df = load_city(str(ROOT / "raw_data" / "nagoya_challengedata.csv"))
    df = label_holidays(df)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"[runner] saved {parquet_path}  ({len(df):,} rows)")


def step1_run_analysis():
    print("\n[runner] === step1: run_analysis.py ===")
    from analysis.run_analysis import main as run_main
    run_main(str(ROOT / "raw_data" / "nagoya_challengedata.csv"))


def step2_poi_hdbscan():
    print("\n[runner] === step2: poi_hdbscan_analysis.py ===")
    from analysis.poi_hdbscan_analysis import main as poi_main
    poi_main()


def step3_personal_trajectory():
    print("\n[runner] === step3: personal_trajectory_analysis.py ===")
    from analysis.personal_trajectory_analysis import main as traj_main
    traj_main()


def copy_figures():
    print("\n[runner] === copying figures ===")
    count = 0
    for src in OUTPUT_FIG.glob("*.png"):
        for dst_dir in [DOCS_FIG, REPORT_IMG]:
            shutil.copy2(src, dst_dir / src.name)
        count += 1
    print(f"[runner] copied {count} figures → docs/figures/ and report/img/")


if __name__ == "__main__":
    step0_create_parquet()
    step1_run_analysis()
    step2_poi_hdbscan()
    step3_personal_trajectory()
    copy_figures()
    print("\n[runner] ALL DONE.")
