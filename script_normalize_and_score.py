#########
# script_normalize_and_score.py
# KCP
# 28/7/24
#########

"""
This is the first script in the recommendation system. This script receives the LLM-Perf data set as an input, and 
returns the dataset with appropriate 'Energy-Performance Score' values added as an output df
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(data):
    """
    Cleans and normalizes the relevant columns in the dataframe.
    
    Input:
    - data: input dataframe containing the LLM-Perf dataset
    Output:
    - dataframe: dataframe with cleaned and normalized variables
    """
    # Clean input df
    # remove rows with all NA values
    data = data.dropna(how='all')
    # Remove * from Open LLM score variable
    data['Open LLM Score (%)'] = data['Open LLM Score (%)'].str.replace('*','').astype(float)

    # normalize columns relevant to the 'Energy-Performance Score' 
    scaler = MinMaxScaler()
    data['Norm_Energy'] = scaler.fit_transform(data[['Energy (tokens/kWh)']])
    data['Norm_LLM_Score'] = scaler.fit_transform(data[['Open LLM Score (%)']])
    data['Norm_Params'] = scaler.fit_transform(data[['Params (B)']])
    data['Norm_Memory'] = scaler.fit_transform(data[['Memory (MB)']])
    # Invert the Memory normalization, because lower memory scores higher
    data['Norm_Memory'] = 1 - data['Norm_Memory']  

    return data

def calculate_energy_performance_score(data):
    """
    Calculates the Energy-Performance Score based on the normalized columns and weights.

    Input:
    - data: input df containing the LLM-Perf data with normalized columns
    Output:
    - dataframe: dataframe with calculated Energy-Performance score
    """
    # define weights for each normalized column
    weights = {
        'Energy': 0.34,
        'LLM_Score': 0.33,
        'Memory': 0.33
    }
    # calculate the weighted sum energy-performance score
    data['Energy_Performance_Score'] = (
        data['Norm_Energy'] * weights['Energy'] +
        data['Norm_LLM_Score'] * weights['LLM_Score'] +
        data['Norm_Memory'] * weights['Memory']
    )

    return data

if __name__ == "__main__":
    # load the LLM performance data from csv file. File must be in same working directory and exactly named
    data = pd.read_csv('llm_perf_data.csv')  # Change file name here if needed
    # normalize data
    data = normalize_data(data)
    # calculate energy performance score 
    data = calculate_energy_performance_score(data)
    # save normalized scored data to a new csv file
    data.to_csv('normalized_scored_data.csv', index=False)

    print("Energy-Performance scored data saved to normalized_scored_data.csv")
