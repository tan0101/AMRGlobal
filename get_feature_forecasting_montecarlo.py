
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.simplefilter('ignore', FutureWarning)

def update_progress(progress: float) -> None:
    """Simple stdout progress bar (0.0 to 1.0)."""
    bar_length = 100
    status = ""
    try:
        progress = float(progress)
    except Exception:
        progress = 0.0
        status = "error: progress var must be float\r\n"

    if progress < 0:
        progress = 0.0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1.0
        status = "Done...\r\n"

    block = int(round(bar_length * progress))
    text = f"\rPercent: [{'#' * block + '-' * (bar_length - block)}] {round(progress * 100, 2)}% {status}"
    sys.stdout.write(text)
    sys.stdout.flush()


def build_indicator_forecast_path(specie: str, class_name: str, indicator_code: str) -> str:
    return (
        "Forecasting/MonteCarloIndicatorForecasts/"
        f"IndicatorForecast_montecarlosimulations_{specie}_{class_name}_Indicator_{indicator_code}.csv"
    )


def compute_and_save_forecast(
    file_indicator: str,
    out_path: str,
    Sims: int,
    intercept: float,
    intercept_stddev: float,
    slope: float,
    slope_stddev: float,
) -> Tuple[str, Optional[str]]:

    try:
        indicator_forecast = pd.read_csv(file_indicator, header=0, index_col=0)

        ratesforecast_intercept = np.random.normal(intercept, intercept_stddev, Sims)
        ratesforecast_slope = np.random.normal(slope, slope_stddev, Sims)

        IF_vals = indicator_forecast.values  

        rates_matrix = IF_vals * ratesforecast_slope[np.newaxis, :] + ratesforecast_intercept[np.newaxis, :]
        np.clip(rates_matrix, 0.0, 1.0, out=rates_matrix)

        df_out = pd.DataFrame(rates_matrix.T, columns=indicator_forecast.index.tolist())

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path)

        return out_path, None
    except Exception as e:
        return out_path, f"{type(e).__name__}: {e}"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python get_feature_forecasting_montecarlo.py <species> [--workers N]")
    specie = sys.argv[1]
    # Optional workers
    workers: Optional[int] = None
    if "--workers" in sys.argv:
        try:
            idx = sys.argv.index("--workers")
            workers = int(sys.argv[idx + 1])
        except Exception:
            raise SystemExit("Invalid --workers argument. Usage: --workers <int>")

    # Number of Monte Carlo simulations
    Sims = 10000

    out_dir = "Results/Forecasting/MonteCarloFeaturesForecasts"
    os.makedirs(out_dir, exist_ok=True)

    indicators_df = pd.read_csv(
        "Results/Correlations/Normalized_indicators.csv",
        index_col=0,
        header=0
    )
    indicator_bounds = pd.read_csv(
        "Results/Forecasting Global/IndicatorForecastBounds.csv",
        index_col=0,
        header=0
    )

    lr_path = f"Results/Forecasting/LinearRegression/LinearRegression_{specie}_permutation.xlsx"
    linear_regression_all = pd.read_excel(lr_path, sheet_name=None, header=0, index_col=None)


    corr_path = f"Results/Correlations/{specie}.xlsx"
    features_rate_all = pd.read_excel(corr_path, sheet_name=None, header=0, index_col=0)

    for class_name, regression_params in linear_regression_all.items():
        print(class_name)

        features_rate_df = features_rate_all.get(class_name)
        if features_rate_df is None or features_rate_df.empty:
            continue

        features_rate_df = (
            features_rate_df
            .dropna(axis=0, how="any")
            .loc[features_rate_df["Num Isolates"] > 5]
            .reset_index(drop=True)
        )
        if features_rate_df.empty:
            continue

        merged_all = features_rate_df.merge(
            indicators_df,
            left_on=["Country Code", "Collection Year"],
            right_on=["Country Code", "Year"],
            how="left",
            suffixes=(None, "_ind")
        )

        years_observed = np.sort(merged_all["Collection Year"].unique())
        if years_observed.size < 5:
            continue
        last_year = int(years_observed.max())

        indicator_validity: Dict[str, Tuple[bool, np.ndarray]] = {}
        for col in indicators_df.columns:

            if col in ("Country Code", "Year"):
                continue
            if col in merged_all.columns:
                vals = merged_all[col].to_numpy()
                mask_valid = ~np.isnan(vals)
                is_valid = (mask_valid.sum() >= 5) and (np.unique(vals[mask_valid]).size >= 2)
                indicator_validity[col] = (is_valid, mask_valid)

        tasks = []
        reg_rows = regression_params.reset_index(drop=True)
        total_rows = len(reg_rows)

        update_progress(0.0)

        with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 1) as executor:
            future_to_k = {}

            for k_idx, row in reg_rows.iterrows():
                indicator = row["Indicator"]
                feature = row["Feature"]

                if indicator not in indicator_bounds.index:
                    update_progress((k_idx + 1) / total_rows)
                    continue
                code = str(indicator_bounds.loc[indicator, "Code"])

                valid_info = indicator_validity.get(indicator)
                if not valid_info or not valid_info[0]:
                    update_progress((k_idx + 1) / total_rows)
                    continue

                file_indicator = build_indicator_forecast_path(specie, class_name, code)
                if not Path(file_indicator).exists():
                    update_progress((k_idx + 1) / total_rows)
                    continue

                try:
                    forecast_years = pd.read_csv(file_indicator, header=0, index_col=0).index.to_numpy()
                except Exception:
                    update_progress((k_idx + 1) / total_rows)
                    continue

                if (forecast_years <= last_year).sum() < 5:
                    update_progress((k_idx + 1) / total_rows)
                    continue

                n = float(row["n"])
                intercept = float(row["intercept"])
                intercept_stderr = float(row["intercept_STDerr"])
                intercept_stddev = np.sqrt(n) * intercept_stderr

                slope = float(row["slope"])
                slope_stderr = float(row["slope_STDerr"])
                slope_stddev = np.sqrt(n) * slope_stderr

                out_path = (
                    f"{out_dir}/FeatureForecast_{specie}_{class_name}_LinearRegressionRow_{k_idx}.csv"
                )

                fut = executor.submit(
                    compute_and_save_forecast,
                    file_indicator=file_indicator,
                    out_path=out_path,
                    Sims=Sims,
                    intercept=intercept,
                    intercept_stddev=intercept_stddev,
                    slope=slope,
                    slope_stddev=slope_stddev,
                )
                future_to_k[fut] = k_idx

            completed = 0
            for fut in as_completed(future_to_k):
                k_done = future_to_k[fut]
                out_path, err = fut.result()
                completed += 1
                update_progress(min(1.0, completed / max(1, len(future_to_k))))
                if err:
                    print(f"\n[WARN] Row {k_done} failed -> {err}")

        update_progress(1.0)
        print()  


if __name__ == "__main__":
    main()
