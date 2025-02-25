import numpy as np
import pandas as pd
import os
import glob
import sys

from pathlib import Path

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

if __name__ == "__main__":
    folder =  "Data"

    # Load AMR Data
    antibiotic_class_df = pd.read_csv("Data/antibiotic_class.csv", header=[0], index_col=[0])

    specie_name = sys.argv[1]

    name_file = "Data/"+specie_name+"_RSI_Class.csv"

    # Load AMR Data
    amr_df = pd.read_csv(folder+"/"+specie_name+"_RSI.csv", header=[0], index_col=[0])

    # Load Metadata
    metadata_df = pd.read_csv(folder+"/"+specie_name+"_metadata.csv", header=[0], index_col=[0])
    metadata_df = metadata_df.loc[amr_df.index,:]

    # Load AMR Class Data
    class_RSI_df = pd.read_csv(name_file, header=[0], index_col=[0])

    # Load Data ARGs:
    data_args_df = pd.read_csv(folder+"/"+specie_name+'_ARGs.csv', header = [0], index_col=[0])
    data_args_df = data_args_df.loc[amr_df.index,:]
    data_args_df[data_args_df>0]=1
    
    # Load Data MGEs:
    data_mge_df = pd.read_csv(folder+"/"+specie_name+'_MGEs.csv', header = [0], index_col=[0])
    data_mge_df = data_mge_df.loc[amr_df.index,:]
    data_mge_df[data_mge_df>0]=1

    # Load Data Plasmid with ARGs:
    data_plasmid_df = pd.read_csv(folder+"/"+specie_name+'_PlasmidARGs.csv', header = [0], index_col=[0])
    data_plasmid_df = data_plasmid_df.loc[amr_df.index,:]
    data_plasmid_df[data_plasmid_df>0]=1  
    
    # Concatenate Data:
    data_comb_df = pd.concat([data_args_df, data_mge_df, data_plasmid_df], axis=1)

    countries_unique = np.unique(metadata_df["Country Code"])
    res_df = pd.DataFrame()

    k = 0
    update_progress(0)
    for count, country in enumerate(countries_unique):
        idx_country = np.where(metadata_df["Country Code"] == country)[0]

        year_unique = np.unique(metadata_df.loc[metadata_df.index[idx_country], "Collection Year"])

        for year in year_unique:
            res_df.loc[k, "Country Code"] = country
            res_df.loc[k, "Collection Year"] = year

            idx_year = np.where(metadata_df["Collection Year"] == year)[0]
            n_isolates = len(np.intersect1d(idx_year, idx_country))

            res_df.loc[k, "Num Isolates"] = n_isolates
            k+=1

        update_progress((count+1)/len(countries_unique))

    if not os.path.exists("Results/Feature Rate"):
        os.makedirs("Results/Feature Rate")

    writer = pd.ExcelWriter("Results/Feature Rate/"+specie_name+".xlsx", engine='xlsxwriter')

    update_progress(0)
    for count, class_name in enumerate(class_RSI_df.columns):
        features_class = []
        for name_antibiotic in amr_df.columns:
            if name_antibiotic not in antibiotic_class_df.index:
                continue
            
            if antibiotic_class_df.loc[name_antibiotic,"Antibiotic Class"] != class_name:
                continue

            if "_" in name_antibiotic:
                name_antibiotic_test = name_antibiotic.replace("_","-")
            else:
                name_antibiotic_test = name_antibiotic

            # Get features
            performance_file = "Results/"+specie_name+" AMR PA/ML/SMOTE_results_"+specie_name+"_"+name_antibiotic_test+".csv"
            
            my_file = Path(performance_file)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue

            df_performance = pd.read_csv(performance_file, header=[0], index_col=[0])
            performance_array = np.array(df_performance.loc["AUC_Mean",:])

            if len(np.where(performance_array > 0.9)[0])==0:
                continue

            # Get features
            features_file = "Results/"+specie_name+" AMR PA/ML/features_"+specie_name+"_"+name_antibiotic_test+".csv"                
            df_features = pd.read_csv(features_file, header=[0], index_col=[0])

            for feat in df_features[df_features.columns[0]]:
                features_class.append(feat)

        features_unique = np.unique(features_class)

        if len(features_unique) == 0:
            continue
                    
        results_df = res_df.copy()

        for count_feat, feat in enumerate(features_unique):
            if count_feat == 0:
                for k in range(len(results_df)):
                    idx_country = np.where(metadata_df["Country Code"] == results_df.loc[k,"Country Code"])[0]
                    idx_year = np.where(metadata_df["Collection Year"] == results_df.loc[k,"Collection Year"])[0]

                    isolate_intersect = np.array(metadata_df.index[np.intersect1d(idx_country, idx_year)])

                    R_S_count = 0
                    R_rate_count = 0
                    for isolate in isolate_intersect:
                        if class_RSI_df.loc[isolate,class_name] == "R" or class_RSI_df.loc[isolate,class_name] == "S":
                            R_S_count+=1
                        
                        if class_RSI_df.loc[isolate,class_name] == "R":
                            R_rate_count+=1

                    if R_S_count == 0:
                        results_df.loc[k,"AMR Rate"] = ""
                    else:
                        results_df.loc[k,"AMR Rate"] = R_rate_count/R_S_count

            for k in range(len(results_df)):
                idx_country = np.where(metadata_df["Country Code"] == results_df.loc[k,"Country Code"])[0]
                idx_year = np.where(metadata_df["Collection Year"] == results_df.loc[k,"Collection Year"])[0]

                isolate_intersect = np.array(metadata_df.index[np.intersect1d(idx_country, idx_year)])

                R_S_count = 0
                R_pres_count = 0
                R_rate_count = 0
                for isolate in isolate_intersect:
                    if class_RSI_df.loc[isolate,class_name] == "R" or class_RSI_df.loc[isolate,class_name] == "S":
                        R_S_count+=1
                    
                    if class_RSI_df.loc[isolate,class_name] == "R" and data_comb_df.loc[isolate,feat] == 1:
                        R_pres_count+=1

                    if class_RSI_df.loc[isolate,class_name] == "R":
                        R_rate_count+=1

                if R_S_count == 0:
                    results_df.loc[k,feat] = ""
                else:
                    results_df.loc[k,feat] = R_pres_count/R_S_count

        results_df.to_excel(writer, sheet_name = class_name, index = True)

        update_progress((count+1)/len(class_RSI_df.columns))

    writer.close() 



