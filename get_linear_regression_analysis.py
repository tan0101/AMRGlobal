
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from joblib import Parallel, delayed

# === CONFIGURATION ===
species = sys.argv[1] if len(sys.argv) > 1 else "genus_species"
new_output = f"Forecasting/LinearRegression/LinearRegression_{species}_permutation.xlsx"
input_path = f"Results/Correlations/{species}.xlsx"
indicators_path = "Results/Correlations/Normalized_indicators.csv"

# Load indicators
indicators_df = pd.read_csv(indicators_path, index_col=[0], header=[0])

# Load feature rate data
feature_rate_df = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")

# Prepare writer for new file
writer = pd.ExcelWriter(new_output, engine="xlsxwriter")

# Precompute indicator columns
indicator_cols = indicators_df.columns[2:]

# Function to process one indicator-feature pair
def process_pair(indicator, feat, anti_df):

    # Merge indicator values into anti_df
    merged_df = anti_df.merge(indicators_df[["Country Code", "Year", indicator]],
                              left_on=["Country Code", "Collection Year"],
                              right_on=["Country Code", "Year"],
                              how="left")
    indicator_array = merged_df[indicator].values

    # Remove NaNs
    valid_idx = ~np.isnan(indicator_array)
    indicator_array = indicator_array[valid_idx]
    rate_array = merged_df[feat].values[valid_idx]
    country_array = merged_df["Country Code"].values[valid_idx]
    year_array = merged_df["Collection Year"].values[valid_idx]

    if len(indicator_array) < 5 or len(np.unique(indicator_array)) == 1 or len(np.unique(rate_array)) == 1:
        return None

    countries = np.unique(country_array)
    years = np.unique(year_array)
    if len(years) < 5:
        return None

    # Pearson correlation
    stats_pearson, pvalue_pearson = pearsonr(rate_array, indicator_array)
    if pvalue_pearson >= 0.05 or abs(stats_pearson) <= 0.5:
        return None

    # Linear regression
    model_df = pd.DataFrame({"Rate": rate_array, "indicator": indicator_array})
    model = smf.ols(formula="Rate ~ indicator", data=model_df).fit()

    # Permutation test
    observed_r2 = model.rsquared
    n_permutations = 1000
    r2_null = []
    for _ in range(n_permutations):
        permuted_df = model_df.copy()
        permuted_df['Rate'] = np.random.permutation(model_df['Rate'])
        perm_model = smf.ols(formula='Rate ~ indicator', data=permuted_df).fit()
        r2_null.append(perm_model.rsquared)
    r2_null = np.array(r2_null)
    p_value = np.mean(r2_null >= observed_r2)

    return {
        "Indicator": indicator,
        "Feature": feat,
        "Pearson r": stats_pearson,
        "Pearson p-value": pvalue_pearson,
        "intercept": model.params.iloc[0],
        "intercept_STDerr": model.bse.iloc[0],
        "intercept_pvalue": model.pvalues.iloc[0],
        "slope": model.params.iloc[1],
        "slope_STDerr": model.bse.iloc[1],
        "slope_pvalue": model.pvalues.iloc[1],
        "Rsquared": model.rsquared,
        "n": len(rate_array),
        "Countries": ', '.join(list(countries)),
        "Num Countries": len(countries),
        "Years": ', '.join(map(str, years)),
        "Num Years": len(years),
        "First Year": np.min(years),
        "Last Year": np.max(years),
        "Permutation_p_val": p_value
    }

# Process each sheet
for class_name, anti_df in feature_rate_df.items():
    anti_df = anti_df.dropna(axis=0, how="any").reset_index(drop=True)
    anti_df = anti_df.loc[anti_df["Num Isolates"] > 5].reset_index(drop=True)

    # Collect all tasks
    tasks = []
    for indicator in indicator_cols:
        for feat in anti_df.columns[3:]:
            if feat in ["Country Code", "Collection Year", "Num Isolates"]:
                continue
            tasks.append((indicator, feat))

    # Parallel processing
    results = Parallel(n_jobs=-1)(delayed(process_pair)(indicator, feat, anti_df) for indicator, feat in tasks)
    results = [r for r in results if r is not None]

    if results:
        new_res_df = pd.DataFrame(results)
        new_res_df.to_excel(writer, sheet_name=class_name, index=False)

writer.close()
