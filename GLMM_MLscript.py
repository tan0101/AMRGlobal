import numpy as np
import pandas as pd
import os
import sys
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import t

NUM_TRIALS = 50
K_FOLDS = 5

def die(msg: str):
    print(msg)
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        die("Usage: python GLMM_MLscript.py <dataset_name> <antibiotic_name>")

    name_dataset = sys.argv[1]
    name_antibiotic = sys.argv[2]

    folder = "Data"
    results_folder = "Results"

    # Load data
    rsi_path = f"{folder}/{name_dataset}_RSI.csv"
    args_path = f"{folder}/{name_dataset}_ARGs.csv"
    mge_path = f"{folder}/{name_dataset}_MGEs.csv"
    plasmid_path = f"{folder}/{name_dataset}_PlasmidARGs.csv"
    metadata_path = f"{folder}/{name_dataset}_metadata.csv"

    for p in [rsi_path, args_path, mge_path, plasmid_path, metadata_path]:
        if not os.path.exists(p):
            die(f"Missing required file: {p}")

    antibiotic_df = pd.read_csv(rsi_path, header=0, index_col=0)
    data_args_df = pd.read_csv(args_path, header=0, index_col=0)
    data_mge_df = pd.read_csv(mge_path, header=0, index_col=0)
    data_plasmid_df = pd.read_csv(plasmid_path, header=0, index_col=0)
    metadata = pd.read_csv(metadata_path, header=0, index_col=0)

    for df in [data_args_df, data_mge_df, data_plasmid_df, metadata]:
        missing = set(antibiotic_df.index) - set(df.index)
        if missing:
            die(f"Metadata/features missing rows present in RSI file. First few missing IDs: {list(missing)[:5]}")
        df.sort_index(inplace=True)

    data_args_df = data_args_df.loc[antibiotic_df.index]
    data_mge_df = data_mge_df.loc[antibiotic_df.index]
    data_plasmid_df = data_plasmid_df.loc[antibiotic_df.index]
    metadata = metadata.loc[antibiotic_df.index]

    data_comb_df = pd.concat([data_args_df, data_mge_df, data_plasmid_df], axis=1)

    target_str = antibiotic_df[name_antibiotic].to_numpy()
    target = np.zeros(len(target_str), dtype=int)
    idx_R = np.where(target_str == 'R')[0]
    idx_S = np.where(target_str == 'S')[0]
    target[idx_R] = 1

    idx_NA = np.where((target_str != 'R') & (target_str != 'S'))[0]
    if len(idx_NA) > 0:
        keep_mask = np.ones(len(target_str), dtype=bool)
        keep_mask[idx_NA] = False
        target = target[keep_mask]
        data_comb_df = data_comb_df.iloc[keep_mask, :]
        metadata = metadata.iloc[keep_mask, :]

    count_class = Counter(target)
    if count_class[0] < 0.1 * len(target) or count_class[1] < 0.1 * len(target):
        die("Insufficient class balance for binary classification (each class should be >=10%).")

    directory = os.path.join(results_folder, f"{name_dataset}_AMR")
    os.makedirs(directory, exist_ok=True)
    
    zv_dir = os.path.join(directory, "Accessory_files")
    os.makedirs(zv_dir, exist_ok=True)
    
    numeric_cols = data_comb_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in data_comb_df.columns if c not in numeric_cols]

    min_presence = 1

    present_counts = (data_comb_df[numeric_cols] > 0).sum(axis=0)
    rare_feats = present_counts[present_counts < min_presence].index.tolist()
    if rare_feats:
        data_comb_df.drop(columns=rare_feats, inplace=True, errors="ignore")
        pd.Series(rare_feats, name=f"Features_present_in_<_{min_presence}_samples").to_csv(
            os.path.join(zv_dir, f"rare_features_lt_{min_presence}_samples_removed.csv"), index=False
        )
    else:
        pd.Series([], dtype=str, name=f"Features_present_in_<_{min_presence}_samples").to_csv(
            os.path.join(zv_dir, f"rare_features_lt_{min_presence}_samples_removed.csv"), index=False
        )

    numeric_cols = data_comb_df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in data_comb_df.columns if c not in numeric_cols]

    variances = data_comb_df[numeric_cols].var(axis=0, ddof=0)
    zero_var_features = variances[variances == 0].index.tolist()

    if zero_var_features:
        data_comb_df.drop(columns=zero_var_features, inplace=True, errors="ignore")
        pd.Series(zero_var_features, name="ZeroVarianceFeatures").to_csv(
        )
    else:
        pd.Series([], dtype=str, name="ZeroVarianceFeatures").to_csv(
            os.path.join(zv_dir, "zero_variance_features_removed.csv"), index=False
        )

    X = data_comb_df.to_numpy(dtype=float)
    exog = np.column_stack([np.ones(len(target)), X])
    exog_names = ["Intercept"] + list(data_comb_df.columns)
    
    if "Country Code" not in metadata.columns or "Host Name" not in metadata.columns or "Collection Year" not in metadata.columns or "PP cluster" not in metadata.columns:
        die("Metadata must contain 'Country Code','Host Name', 'Collection Year' and 'PP cluster' columns (case-sensitive).")

    # Categorical variables
    country_series = metadata["Country Code"].astype("category")
    host = metadata["Host Name"].astype("category")
    year = metadata["Collection Year"].astype("category")
    ppcluster = metadata["PP cluster"].astype("category")

    # Variance component design matrices
    country_dummies = csr_matrix(pd.get_dummies(country_series, drop_first=False).to_numpy(dtype=float))
    host_dummies = csr_matrix(pd.get_dummies(host, drop_first=False).to_numpy(dtype=float))
    year_dummies = csr_matrix(pd.get_dummies(year, drop_first=False).to_numpy(dtype=float))
    ppcluster_dummies = csr_matrix(pd.get_dummies(ppcluster, drop_first=False).to_numpy(dtype=float))

    exog_vc = hstack([country_dummies, host_dummies, year_dummies, ppcluster_dummies], format="csr")
    ident = np.array([0] * country_dummies.shape[1]
                     + [1] * host_dummies.shape[1]
                     + [2] * year_dummies.shape[1]
                     + [3] * ppcluster_dummies.shape[1])

    vcp_names = ["country", "host", "year", "ppcluster"]
    vc_names = ([f"country:{lev}" for lev in country_series.cat.categories] +
                [f"host:{lev}" for lev in host.cat.categories] +
                [f"year:{lev}" for lev in year.cat.categories] +
                [f"ppcluster:{lev}" for lev in ppcluster.cat.categories])

    # Accumulators
    all_metrics = []
    coef_runs = []
    or_runs = []
    vcomp_runs = []
    re_runs = []

    # ===== Cross-validation across NUM_TRIALS =====
    for run in range(NUM_TRIALS):
        print(f"Run {run+1}/{NUM_TRIALS}")
        run_dir = os.path.join(directory, f"Run_{run+1}")
        os.makedirs(run_dir, exist_ok=True)

        sgkf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=run)
        fold_idx = 0
        for train_idx, test_idx in sgkf.split(X=exog, y=target):
            fold_idx += 1
            y_train, y_test = target[train_idx], target[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print(f"Skipping fold {fold_idx}: fewer than 2 classes.")
                continue

            fold_dir = os.path.join(run_dir, f"Fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            endog_train, endog_test = y_train, y_test
            exog_train, exog_test = exog[train_idx, :], exog[test_idx, :]
            exog_vc_train, exog_vc_test = exog_vc[train_idx, :], exog_vc[test_idx, :]

            # Fit mixed model
            model = BinomialBayesMixedGLM(
                endog=endog_train,
                exog=exog_train,
                exog_vc=exog_vc_train,
                ident=ident,
                vcp_p=0.5,
                fe_p=2.0,
                fep_names=exog_names,
                vcp_names=vcp_names,
                vc_names=vc_names
            )
            result = model.fit_vb()

            # Predict & metrics on test
            pred_probs_test = result.predict(exog_test)
            pred_labels_test = (pred_probs_test >= 0.5).astype(int)

            auc = roc_auc_score(endog_test, pred_probs_test)
            pr_auc = average_precision_score(endog_test, pred_probs_test)
            acc = accuracy_score(endog_test, pred_labels_test)
            cm = confusion_matrix(endog_test, pred_labels_test, labels=[0, 1])
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
            prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan

            all_metrics.append({
                "Run": run + 1,
                "Fold": fold_idx,
                "AUC": auc,
                "PR_AUC": pr_auc,
                "Accuracy": acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "Precision": prec
            })

            # Save per-fold metrics & predictions
            metrics_df = pd.DataFrame({
                "Metric": ["AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "Precision"],
                "Value": [auc, pr_auc, acc, sens, spec, prec]
            })
            metrics_df.to_csv(os.path.join(fold_dir, f"{name_dataset}_{name_antibiotic}_metrics.csv"), index=False)

            pd.DataFrame({
                "Sample": data_comb_df.index[test_idx],
                "True_Label": endog_test,
                "Predicted_Label": pred_labels_test,
                "Predicted_Probability": pred_probs_test
            }).to_csv(os.path.join(fold_dir, f"{name_dataset}_{name_antibiotic}_predictions.csv"), index=False)

            # Fixed-effects coefficients
            coef_df = pd.DataFrame({"Feature": exog_names, "Coefficient": result.fe_mean, "Posterior_SD": result.fe_sd})
            coef_df["Run"] = run + 1
            coef_df["Fold"] = fold_idx
            coef_df.to_csv(os.path.join(fold_dir, f"{name_dataset}_{name_antibiotic}coefficients.csv"), index=False)

            # Odds ratios (+ 95% CI)
            odds_ratios = np.exp(result.fe_mean)
            lower_ci = np.exp(result.fe_mean - 1.96 * result.fe_sd)
            upper_ci = np.exp(result.fe_mean + 1.96 * result.fe_sd)
            or_df = pd.DataFrame({
                "Feature": exog_names,
                "Coefficient": result.fe_mean,
                "Odds_Ratio": odds_ratios,
                "OR_95CI_Lower": lower_ci,
                "OR_95CI_Upper": upper_ci,
                "Run": run + 1,
                "Fold": fold_idx
            })
            or_df.to_csv(os.path.join(fold_dir, f"{name_dataset}_{name_antibiotic}odds_ratios.csv"), index=False)

            # Variance components (SDs)
            vc_sd = np.exp(result.vcp_mean)
            vcomp_df = pd.DataFrame({"Random_Effect": vcp_names, "SD": vc_sd, "Run": run + 1, "Fold": fold_idx})
            vcomp_df.to_csv(os.path.join(fold_dir, f"{name_dataset}_{name_antibiotic}variance_components.csv"), index=False)

            # Random effects by level
            re_df = result.random_effects()
            if isinstance(re_df, pd.Series):
                re_df = re_df.to_frame(name='value')
            re_df["Run"] = run + 1
            re_df["Fold"] = fold_idx
            re_df.to_csv(os.path.join(fold_dir, name_dataset + '_' + name_antibiotic + '_' + "random_effects_by_level.csv"))

            # Accumulate for cross-run summaries
            coef_runs.append(coef_df)
            or_runs.append(or_df)
            vcomp_runs.append(vcomp_df)
            re_runs.append(re_df)

        # Update running summary metrics after each run
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_csv(os.path.join(directory, name_dataset + '_' + name_antibiotic + '_' + "summary_metrics.csv"), index=False)
        print("Run completed. Summary (so far) saved.")

    print("All runs completed. Summary saved.")

    # Aggregate results
    # Overall metrics (mean & std across all valid folds/trials)
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        metric_names = ["AUC", "PR_AUC", "Accuracy", "Sensitivity", "Specificity", "Precision"]
        overall_mean = summary_df[metric_names].mean(numeric_only=True)
        overall_std = summary_df[metric_names].std(ddof=1, numeric_only=True)
        overall_df = pd.DataFrame({
            "Metric": metric_names,
            "Mean": [overall_mean[m] for m in metric_names],
            "Std": [overall_std[m] for m in metric_names]
        })
        overall_df.to_csv(os.path.join(directory, name_dataset + '_' + name_antibiotic + '_' + "summary_metrics_overall.csv"), index=False)

    # Coefficients averaged across folds/trials
    if coef_runs:
        
        coef_all = pd.concat(coef_runs, axis=0, ignore_index=True)
        # Average coefficient and posterior SD per feature
        coef_summary = (
            coef_all.groupby("Feature", as_index=False)[["Coefficient", "Posterior_SD"]]
            .mean()
        )
        coef_summary.to_csv(os.path.join(directory, f"{name_dataset}_{name_antibiotic}_summary_coefficients_with_uncertainty.csv"), index=False)

    # Odds ratios averaged across folds/trials
    if or_runs:
        or_all = pd.concat(or_runs, axis=0, ignore_index=True)
        agg_cols = ["Coefficient", "Odds_Ratio", "OR_95CI_Lower", "OR_95CI_Upper"]
        or_mean = (or_all
                   .groupby("Feature", as_index=False)[agg_cols]
                   .mean())
        or_mean.to_csv(os.path.join(directory, name_dataset + '_' + name_antibiotic + '_' + "summary_odds_ratios_mean.csv"), index=False)

    # Variance components averaged across folds/trials
    if vcomp_runs:
        vcomp_all = pd.concat(vcomp_runs, axis=0, ignore_index=True)
        vcomp_mean = (vcomp_all
                      .groupby("Random_Effect", as_index=False)["SD"]
                      .mean())
        vcomp_mean.to_csv(os.path.join(directory, name_dataset + '_' + name_antibiotic + '_' + "summary_variance_components_mean.csv"), index=False)

    # Random effects by level averaged across folds/trials
    if re_runs:
        re_all = pd.concat(re_runs, axis=0)
        if isinstance(re_all, pd.Series):
            re_all = re_all.to_frame(name='value')
        # Select numeric columns (exclude Run/Fold)
        numeric_cols = [c for c in re_all.select_dtypes(include=['number']).columns if c not in ("Run", "Fold")]
        if len(numeric_cols) > 0:
            re_mean = re_all.groupby(re_all.index)[numeric_cols].mean()
            re_mean.to_csv(os.path.join(directory, name_dataset + '_' + name_antibiotic + '_' + "summary_random_effects_by_level_mean.csv"))

    print("Averaged summaries across folds and trials saved.")

# Final refit on FULL DATA (for inference ONLY; do NOT use for performance)
print("Refitting final model on the full dataset for inference...")

# Full-data endog/exog (already constructed earlier)
endog_full = target
exog_full = exog            # includes Intercept + all fixed-effect columns
exog_vc_full = exog_vc      # stacked variance-component design (country/host/year/ppcluster)

final_model = BinomialBayesMixedGLM(
    endog=endog_full,
    exog=exog_full,
    exog_vc=exog_vc_full,
    ident=ident,
    vcp_p=0.5,
    fe_p=2.0,
    fep_names=exog_names,
    vcp_names=vcp_names,
    vc_names=vc_names
)
final_result = final_model.fit_vb()

# Output directory for full-data inference artifacts
final_dir = os.path.join(directory, "Final_FullData_Model")
os.makedirs(final_dir, exist_ok=True)

final_coef_df = pd.DataFrame({
    "Feature": exog_names,
    "Coefficient": final_result.fe_mean,
    "Posterior_SD": final_result.fe_sd
})
final_coef_df.to_csv(os.path.join(final_dir, f"{name_dataset}_{name_antibiotic}_FULL_coefficients.csv"),
                     index=False)

# ---------- Odds ratios with 95% CI ----------

coef_all = pd.concat(coef_runs, ignore_index=True)

cv_rows = []
for feat, grp in coef_all.groupby("Feature"):
    vals = grp["Coefficient"].astype(float).dropna().to_numpy()
    m = len(vals)

    if m < 2:
        cv_rows.append({
            "Feature": feat,
            "logOR_CV_mean": np.nan if m == 0 else float(vals.mean()),
            "logOR_CV_CI_lower": np.nan,
            "logOR_CI_CI_upper": np.nan,
        })
        continue

    mean_log = float(vals.mean())
    s = float(np.std(vals, ddof=1))
    se = s / np.sqrt(m)
    tcrit = float(t.ppf(0.975, df=m - 1))

    cv_rows.append({
        "Feature": feat,
        "logOR_CV_mean": mean_log,
        "logOR_CV_CI_lower": mean_log - tcrit * se,
        "logOR_CV_CI_upper": mean_log + tcrit * se,
    })

cv_df = pd.DataFrame(cv_rows)

final_or_df = pd.DataFrame({
    "Feature": exog_names,
    "Coefficient": final_result.fe_mean,
    "Odds_Ratio": np.exp(final_result.fe_mean),
})

merged = final_or_df.merge(cv_df, on="Feature", how="left")

merged["OR_95CI_Lower"] = np.exp(merged["logOR_CV_CI_lower"])
merged["OR_95CI_Upper"] = np.exp(merged["logOR_CV_CI_upper"])

merged.to_csv(
    os.path.join(final_dir, f"{name_dataset}_{name_antibiotic}_FULL_odds_ratios_CVbasedCI.csv"),
    index=False
)


final_vc_sd = np.exp(final_result.vcp_mean)
final_vcomp_df = pd.DataFrame({
    "Random_Effect": vcp_names,
    "SD": final_vc_sd
})
final_vcomp_df.to_csv(os.path.join(final_dir, f"{name_dataset}_{name_antibiotic}_FULL_variance_components.csv"),
                      index=False)

# ---------- Random effects by level ----------
final_re_df = final_result.random_effects()
if isinstance(final_re_df, pd.Series):
    final_re_df = final_re_df.to_frame(name="value")
final_re_df.to_csv(os.path.join(final_dir, f"{name_dataset}_{name_antibiotic}_FULL_random_effects_by_level.csv"))


# ---------- ICCs per random-effect family ----------
# Logistic residual variance
latent_var = (np.pi ** 2) / 3.0

# final_vc_sd is SD per family (already length = number of families)
# vcp_names identifies those families (['country','host','year','ppcluster'])

vc_vars = final_vc_sd ** 2        # variance per family (1 per RE type)
total_var = np.sum(vc_vars) + latent_var

icc_rows = []
for fam_name, var_k in zip(vcp_names, vc_vars):
    icc_k = var_k / total_var
    icc_rows.append({
        "Random_Effect_Family": fam_name,
        "Variance": var_k,
        "ICC": icc_k
    })

final_icc_df = pd.DataFrame(icc_rows)
final_icc_df.to_csv(os.path.join(final_dir, f"{name_dataset}_{name_antibiotic}_FULL_ICCs.csv"),
                    index=False)


print("Final full-data model refit complete. Inference artifacts written to:", final_dir)


# --- Summaries across all trials ---


# # Bootstrap helper for mean CI
# def bootstrap_ci_mean(values, B=1000, alpha=0.05, random_state=42):
#     vals = pd.Series(values).dropna().to_numpy()
#     n = len(vals)
#     if n == 0:
#         return (np.nan, np.nan)
#     rng = np.random.default_rng(random_state)
#     means = np.empty(B, dtype=float)
#     for b in range(B):
#         resample = rng.choice(vals, size=n, replace=True)
#         means[b] = np.mean(resample)
#     lower = float(np.quantile(means, alpha/2))
#     upper = float(np.quantile(means, 1 - alpha/2))
#     return (lower, upper)

# if coef_runs:
#     coef_all = pd.concat(coef_runs, axis=0, ignore_index=True)
#     rows = []
#     for feat, grp in coef_all.groupby('Feature'):
#         vals = grp['Coefficient'].astype(float)
#         m = float(vals.mean())
#         s = float(vals.std(ddof=1))
#         lo, hi = bootstrap_ci_mean(vals)
#         rows.append({
#             'Feature': feat,
#             'Coefficient_mean': m,
#             'Coefficient_std': s,
#             'Coefficient_mean_CI_lower': lo,
#             'Coefficient_mean_CI_upper': hi
#         })
#     coef_summary = pd.DataFrame(rows)
#     coef_summary.to_csv(os.path.join(directory, name_dataset +'_'+ name_antibiotic+'_'+'summary_coefficients.csv'), index=False)


# if or_runs:
#     or_all = pd.concat(or_runs, axis=0, ignore_index=True)
#     rows = []
#     for feat, grp in or_all.groupby('Feature'):
#         or_vals = grp['Odds_Ratio'].astype(float)
#         l_vals = grp['OR_95CI_Lower'].astype(float)
#         u_vals = grp['OR_95CI_Upper'].astype(float)
#         or_m, or_s = float(or_vals.mean()), float(or_vals.std(ddof=1))
#         l_m, l_s = float(l_vals.mean()), float(l_vals.std(ddof=1))
#         u_m, u_s = float(u_vals.mean()), float(u_vals.std(ddof=1))
#         or_lo, or_hi = bootstrap_ci_mean(or_vals)
#         l_lo, l_hi = bootstrap_ci_mean(l_vals)
#         u_lo, u_hi = bootstrap_ci_mean(u_vals)
#         rows.append({
#             'Feature': feat,
#             'Odds_Ratio_mean': or_m,
#             'Odds_Ratio_std': or_s,
#             'Odds_Ratio_mean_CI_lower': or_lo,
#             'Odds_Ratio_mean_CI_upper': or_hi,
#             'OR_95CI_Lower_mean': l_m,
#             'OR_95CI_Lower_std': l_s,
#             'OR_95CI_Lower_mean_CI_lower': l_lo,
#             'OR_95CI_Lower_mean_CI_upper': l_hi,
#             'OR_95CI_Upper_mean': u_m,
#             'OR_95CI_Upper_std': u_s,
#             'OR_95CI_Upper_mean_CI_lower': u_lo,
#             'OR_95CI_Upper_mean_CI_upper': u_hi
#         })
#     or_summary = pd.DataFrame(rows)
#     or_summary.to_csv(os.path.join(directory, name_dataset +'_'+ name_antibiotic+'_'+'summary_odds_ratios.csv'), index=False)

# if vcomp_runs:
#     vcomp_all = pd.concat(vcomp_runs, axis=0, ignore_index=True)
#     rows = []
#     for re_name, grp in vcomp_all.groupby('Random_Effect'):
#         vals = grp['SD'].astype(float)
#         m = float(vals.mean())
#         s = float(vals.std(ddof=1))
#         lo, hi = bootstrap_ci_mean(vals)
#         rows.append({
#             'Random_Effect': re_name,
#             'SD_mean': m,
#             'SD_std': s,
#             'SD_mean_CI_lower': lo,
#             'SD_mean_CI_upper': hi
#         })
#     vcomp_summary = pd.DataFrame(rows)
#     vcomp_summary.to_csv(os.path.join(directory, name_dataset +'_'+ name_antibiotic+'_'+'summary_variance_components.csv'), index=False)

# if re_runs:
#     re_all = pd.concat(re_runs, axis=0)
#     if isinstance(re_all, pd.Series):
#         re_all = re_all.to_frame(name='value')
#     numeric_cols = re_all.select_dtypes(include=['number']).columns

#     re_median = re_all.groupby(re_all.index)[numeric_cols].median()
  
#     re_q25 = re_all.groupby(re_all.index)[numeric_cols].quantile(0.25)
#     re_q75 = re_all.groupby(re_all.index)[numeric_cols].quantile(0.75)
#     re_median.to_csv(os.path.join(directory, name_dataset +'_'+ name_antibiotic+'_'+'summary_random_effects_by_level_median.csv'))

#     re_iqr_lower = re_q25.copy()
#     re_iqr_lower.columns = [f"{c}_IQR_lower" for c in re_iqr_lower.columns]
#     re_iqr_upper = re_q75.copy()
#     re_iqr_upper.columns = [f"{c}_IQR_upper" for c in re_iqr_upper.columns]
#     re_iqr = re_iqr_lower.join(re_iqr_upper, how='outer')
#     re_iqr.to_csv(os.path.join(directory, name_dataset +'_'+ name_antibiotic+'_'+'summary_random_effects_by_level_IQR.csv'))
