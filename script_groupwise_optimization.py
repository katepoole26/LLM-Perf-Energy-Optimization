#########
# script_groupwise_optimization.py
# KCP
# 28/7/24
#########

"""
This is the second script in the system. This script receives the dataframe from script_normalize_and_score.py 
with the Energy_Performance_Score variable as input. Then, it prepares the data and performs the group-wise 
optimization. The output of this script is the optimal_mem_solutions df csv.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(data):
    """
    Encodes categorical features using LabelEncoder and returns the updated dataframe.

    Input:
    - data: input df containing the LLM performance data
    Output: 
    - dataframe: df with encoded categorical variables
    - dictionary: dict mapping column names to appropriate LabelEncoders
    """
    # dict to store encoders
    column_mappings = {
        'Backend': LabelEncoder(),
        'Precision': LabelEncoder(),
        'Quantization': LabelEncoder(),
        'Attention': LabelEncoder(),
        'Kernel': LabelEncoder()
    }
    # fit and transform each categorical variable using its respective label encoder
    for column, encoder in column_mappings.items():
        data[column] = encoder.fit_transform(data[column])

    return data, column_mappings

def create_memory_group(data):
    """
    Creates a new column 'Memory Group' to categorize the 'Memory (MB)' values into increments of 1000.

    Input:
    - data: input df containing LLM-Perf data
    Output: 
    - dataframe: df with new memory group field
    """
    # categorize Memory variable into increments of 1000 (decided based on inspection)
    data['Memory Group'] = (data['Memory (MB)'] // 1000) * 1000
    
    return data

def find_optimal_solutions(data):
    """
    Finds the optimal combination for maximum 'Energy_Performance_Score' for each memory group.

    Input:
    - data: input df containing LLM-Perf data
    Output: 
    - dataframe: df with optimal solution for each memory quantile 
    """
    # group data by memory group
    grouped_mem_data = data.groupby('Memory Group')

    optimal_mem_solutions = {}
    # for each memory group
    for mem, subgroup in grouped_mem_data:
        # index of the row with the max Energy-Performance score 
        max_energy_idx = subgroup['Energy_Performance_Score'].idxmax()
        # retrieve row with max Energy-Performance score
        optimal_combination = subgroup.loc[max_energy_idx]
        # store optimal configurations in dictionary
        optimal_mem_solutions[mem] = optimal_combination

    # convert dictionary into transposed df
    return pd.DataFrame(optimal_mem_solutions).T

if __name__ == "__main__":
     # Load the normalized and scored data
    data = pd.read_csv('normalized_scored_data.csv') 
    # encode categorical features and get column mappings
    data, column_mappings = encode_categorical_features(data)
    # create memory group column
    data = create_memory_group(data)
    # final optimal solution for each memory group
    optimal_solutions_df = find_optimal_solutions(data)

    # extract all columns to the left of 'Norm_Energy' plus the 'Memory Group' variables 
    norm_energy_index = data.columns.get_loc('Norm_Energy')
    columns_to_extract = data.columns[:norm_energy_index].tolist() + ['Memory Group', 'Energy_Performance_Score']
    optimal_solutions_df = optimal_solutions_df[columns_to_extract]

    # inverse transform encoded features
    for column, encoder in column_mappings.items():
        if column in optimal_solutions_df.columns:
            optimal_solutions_df[column] = encoder.inverse_transform(optimal_solutions_df[column].astype(int))

    # reset index and rename columns for clarity
    optimal_solutions_df.reset_index(inplace=True)
    optimal_solutions_df.rename(columns={'index': 'Memory Group'}, inplace=True)

    # save optimal solutions to a csv file
    optimal_solutions_df.to_csv('optimal_mem_solutions.csv', index=False)

    print("Optimal solutions saved to optimal_mem_solutions.csv")

