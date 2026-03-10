
import pandas as pd
import numpy as np
import os
import glob
from statsmodels.tsa.stattools import adfuller, kpss
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import repeat
from pathlib import Path


def add_unnamed_index_column_to_workbook(xlsx_path: str) -> str:
        if not os.path.isfile(xlsx_path):
            raise FileNotFoundError(f"Input file not found: {xlsx_path}")

        p = Path(xlsx_path)
        if p.suffix.lower() != ".xlsx":
            pass
        new_path = str(p.with_name(f"{p.stem}_with_index{p.suffix}"))
        sheets = pd.read_excel(xlsx_path, sheet_name=None, header=0, engine="openpyxl")

        updated = {}
        for sheet_name, df in sheets.items():
            seq = pd.Series(range(len(df)), index=df.index)
            col = "Unnamed"

            if col in df.columns:
                df[col] = seq
                df = df[[col] + [c for c in df.columns if c != col]]
            else:
                df.insert(0, col, seq)

            updated[sheet_name] = df

        with pd.ExcelWriter(new_path, engine="openpyxl", mode="w") as writer:
            for sheet_name, df_out in updated.items():
                df_out.to_excel(writer, sheet_name=sheet_name, index=False)

        return new_path
    
def analyze_file(filename, indicator_bounds):
    
    # -------- Extract metadata from filename --------
    split_filename = filename.split("_")
    specie = f"{split_filename[1]}_{split_filename[2]}"
    class_name = split_filename[3]
    row = int(split_filename[-1].split(".csv")[0])

    # -------- Load regression parameters (once per task) --------
    
    reg_params = pd.read_excel(
        f"Results/Forecasting/LinearRegression/LinearRegression_{specie}_permutation_with_index.xlsx",
        sheet_name=class_name, header=0, index_col=0
    )
    indicator = reg_params.loc[row, "Indicator"]

    result = {
        "Filename": filename,
        "Specie": specie,
        "Antibiotic Class": class_name,
        "Indicator": reg_params.loc[row, "Indicator"],
        "Feature": reg_params.loc[row, "Feature"],
        "Pearson r": reg_params.loc[row, "Pearson r"],
        "Pearson p-value": reg_params.loc[row, "Pearson p-value"],
        "n": reg_params.loc[row, "n"],
        "Countries": reg_params.loc[row, "Countries"],
        "Num Countries": reg_params.loc[row, "Num Countries"],
        "Years": reg_params.loc[row, "Years"],
        "Num Years": reg_params.loc[row, "Num Years"],
        "First Year": reg_params.loc[row, "First Year"],
        "Last Year": reg_params.loc[row, "Last Year"],
    }


    def stationarity_tests(series, prefix):
        levels = ["1%", "5%", "10%"]  

        def lvl_label(k):  
            return f"{int(100 - int(k[:-1]))}%"

        if len(np.unique(series)) <= 1:
            result[f"{prefix} ADF Test statistic"] = ""
            result[f"{prefix} ADF p-value"] = ""
            for k in levels:
                L = lvl_label(k)
                result[f"{prefix} ADF Confidence Level {L}"] = ""
                result[f"{prefix} ADF Stationary {L}"] = ""
            result[f"{prefix} KPSS Test statistic"] = ""
            result[f"{prefix} KPSS p-value"] = ""
            for k in levels:
                L = lvl_label(k)
                result[f"{prefix} KPSS Confidence Level {L}"] = ""
                result[f"{prefix} KPSS Stationary {L}"] = ""
            for k in levels:
                L = lvl_label(k)
                result[f"{prefix} Result {L}"] = "Constant"
                result[f"{prefix} Trend {L}"] = ""
            return

        adf_stat, adf_p, _, _, adf_crit, *_ = adfuller(series, autolag='AIC')
        kpss_stat, kpss_p, _, kpss_crit = kpss(series, regression="c", nlags="auto")

        result[f"{prefix} ADF Test statistic"] = f"{adf_stat:.3f}"
        result[f"{prefix} ADF p-value"] = f"{adf_p:.3f}"
        result[f"{prefix} KPSS Test statistic"] = f"{kpss_stat:.3f}"
        result[f"{prefix} KPSS p-value"] = f"{kpss_p:.3f}"

        for k in levels:
            if k not in adf_crit or k not in kpss_crit:
                continue
            L = lvl_label(k)

            # ADF
            cv_adf = adf_crit[k]
            result[f"{prefix} ADF Confidence Level {L}"] = cv_adf
            result[f"{prefix} ADF Stationary {L}"] = "No" if cv_adf < adf_stat else "Yes"

            # KPSS
            cv_kpss = kpss_crit[k]
            result[f"{prefix} KPSS Confidence Level {L}"] = cv_kpss
            result[f"{prefix} KPSS Stationary {L}"] = "No" if cv_kpss < kpss_stat else "Yes"

        first_val = series.iloc[0]
        last_val = series.iloc[-1]

        for k in levels:
            L = lvl_label(k)
            adf_flag = result.get(f"{prefix} ADF Stationary {L}", "")
            kpss_flag = result.get(f"{prefix} KPSS Stationary {L}", "")

            if adf_flag == "No" and kpss_flag == "No":
                combined = "Not stationary"
            elif adf_flag == "No" and kpss_flag == "Yes":
                combined = "Trend stationary"
            elif adf_flag == "Yes" and kpss_flag == "No":
                combined = "Difference stationary"
            elif adf_flag == "Yes" and kpss_flag == "Yes":
                combined = "Stationary"
            else:
                combined = ""

            result[f"{prefix} Result {L}"] = combined

            if combined in ("Not stationary", "Difference stationary", "Trend stationary"):
                if first_val > last_val:
                    trend = "Decreasing"
                elif first_val < last_val:
                    trend = "Increasing"
                elif first_val == last_val:
                    trend = "Check!"
                else:
                    trend = ""
            else:
                trend = ""

            result[f"{prefix} Trend {L}"] = trend

    feature_df = pd.read_csv(
        f"Results/Forecasting/MonteCarloFeaturesForecasts/{filename}",
        header=0, index_col=0
    )
    feature_series = feature_df.median(axis=0)
    stationarity_tests(feature_series, "Feature")

    indicator_file = (
        f"Results/Forecasting/MonteCarloIndicatorForecasts/"
        f"IndicatorForecast_montecarlosimulations_{specie}_{class_name}_Indicator_{indicator_bounds.loc[indicator,'Code']}.csv"
    )

    try:
        indicator_df = pd.read_csv(indicator_file, header=0, index_col=0).transpose()
        indicator_series = indicator_df.median(axis=0)
        stationarity_tests(indicator_series, "Indicator")
    except FileNotFoundError:
        for suffix in [
            "ADF Test statistic", "ADF p-value", "KPSS Test statistic", "KPSS p-value"
        ]:
            result[f"Indicator {suffix}"] = ""

        for L in ["99%", "95%", "90%"]:
            result[f"Indicator ADF Confidence Level {L}"] = ""
            result[f"Indicator ADF Stationary {L}"] = ""
            result[f"Indicator KPSS Confidence Level {L}"] = ""
            result[f"Indicator KPSS Stationary {L}"] = ""
            result[f"Indicator Result {L}"] = ""
            result[f"Indicator Trend {L}"] = ""

    return result


if __name__ == "__main__":

    def ensure_with_index_workbook(species_list):
        base_dir = "Results/Forecasting/LinearRegression"
        for specie in species_list:
            src = os.path.join(base_dir, f"LinearRegression_{specie}_permutationtest.xlsx")
            dst = os.path.join(base_dir, f"LinearRegression_{specie}_permutationtest_with_index.xlsx")

            if not os.path.isfile(dst):
                add_unnamed_index_column_to_workbook(src)

    # Load Indicator Bounds
    indicator_bounds = pd.read_csv(
        'Results/Forecasting/IndicatorForecastBounds.csv',
        index_col=0, header=0
    )

    directory = "Results/Forecasting/MonteCarloFeaturesForecasts"
    files_array = glob.glob(os.path.join(directory, "*.csv"))
    basenames = [os.path.basename(f) for f in files_array]
    species = set()
    for filename in basenames:
        parts = filename.split("_")
        specie = f"{parts[1]}_{parts[2]}"
        species.add(specie)

    ensure_with_index_workbook(species)
    results = []

    for filename in tqdm(basenames, desc="Processing files (sequential)"):
        try:
            res = analyze_file(filename, indicator_bounds)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed on file: {filename}")
            print("Exception:", e)
            continue  # skip to next file

    trend_df = pd.DataFrame(results)
    trend_df.to_csv(
        "Results/Forecasting/Trend_Analysis.csv",
        index=False
    )
    print("\nCompleted successfully.")
