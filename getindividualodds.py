import os
import pandas as pd
import sys

species = sys.argv[1]
antibiotic = sys.argv[2]

parent_dir = f"Results/{species}_AMR/"
csv_name_suffix = f"{species}_{antibiotic}odds_ratios.csv"
n_runs = 50

all_run_avgs = []

for i in range(1, n_runs + 1):
    run_folder = os.path.join(parent_dir, f"Run_{i}")
    fold_dfs = []

    for fold in range(1, 6):
        fold_folder = os.path.join(run_folder, f"Fold_{fold}")
        if not os.path.isdir(fold_folder):
            print(f"Warning: Missing {fold_folder}")
            continue

        csv_files = [f for f in os.listdir(fold_folder) if f.endswith(csv_name_suffix)]
        if len(csv_files) != 1:
            print(f"Warning: Found {len(csv_files)} matching files in {fold_folder}")
            continue

        csv_path = os.path.join(fold_folder, csv_files[0])
        df = pd.read_csv(csv_path)

        df = df.set_index("Feature")["Odds_Ratio"]
        fold_dfs.append(df)

    if len(fold_dfs) == 0:
        print(f"Warning: No fold data for Run_{i}")
        continue

    run_avg = pd.concat(fold_dfs, axis=1).mean(axis=1)
    run_avg = run_avg.rename(f"Run_{i}")

    all_run_avgs.append(run_avg)

merged = pd.concat(all_run_avgs, axis=1)

merged["Species"] = species
merged["Antibiotic"] = antibiotic

run_cols = [c for c in merged.columns if c.startswith("Run_")]

# Calculate % of runs where OR > 1
merged["Percent_runs_OR_gt_1"] = (merged[run_cols] > 1).mean(axis=1) * 100

output_path = os.path.join(parent_dir, f"{species}_{antibiotic}_combined_odds_ratios.csv")
merged.to_csv(output_path)

print(f"Combined table saved to: {output_path}")
