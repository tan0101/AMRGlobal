import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import MinMaxScaler

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
    # Load indicators
    indicators_df = pd.read_csv("Data/All indicators(raw).csv", index_col=[0], header=[0])

    results_df = indicators_df.copy()

    countries_array = np.unique(indicators_df["Country Code"])
    update_progress(0)
    for count, indicator in enumerate(indicators_df.columns[2:]):
        for country in countries_array:
            idx = np.where(indicators_df["Country Code"] == country)[0]
            data_array = np.array(indicators_df.loc[idx,indicator])

            if len(np.where(np.isnan(data_array) == False)[0]) > 0:
                scaler = MinMaxScaler()

                data_array = scaler.fit_transform(data_array.reshape(-1,1))
                data_array = data_array.ravel()

                results_df.loc[idx,indicator] = data_array
        
        update_progress((count+1)/len(indicators_df.columns[2:]))

    results_df.to_csv("Results/Normalized_indicators.csv")

        