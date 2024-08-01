#########
# script_recommendation_system.py
# KCP
# 28/7/24
#########

"""
This is the last script in this system. This script recieves as input a new LLM-Perf evaluated observation and the 
optimal_mem_solutions from the script_groupwise_optimization.py script. As output, it returns the optimally configured
model, within the same memory quantile.
"""

import pandas as pd

def recommend_optimized_observation(new_entry, optimal_mem_solutions_df):
    """
    Recommends the optimized observation for a new LLM-Perf data entry based on its memory quantile.

    Input:
    - new_entry: single row df with new LLM-Perf evaulation
    - optimal_mem_solutions_df: df with optimal solutions by subgroup, from script_groupwise_optimization.py
    Output:
    - dataframe: dataframe containing the recommended observation
    """
    # grab memory value for new entry
    memory_mb = new_entry['Memory (MB)'].values[0]
    # categorize in 1000 increments
    memory_group = (memory_mb // 1000) * 1000

    # Retrieve the optimized observation for the memory group
    if memory_group in optimal_mem_solutions_df['Memory Group'].values:
        # if found, return optimal observation
        recommendation = optimal_mem_solutions_df[optimal_mem_solutions_df['Memory Group'] == memory_group]
    else:
        # Handle case where memory group is not found, use closest available group 
        closest_group_idx = (optimal_mem_solutions_df['Memory Group'] - memory_group).abs().idxmin()
        recommendation = optimal_mem_solutions_df.iloc[[closest_group_idx]]

    return recommendation

"""
Remaining code in this script provides an example usage to interact with on the command line
True implementation will NOT include this segment
"""
if __name__ == "__main__":
    # read the optimal solutions dataframe
    optimal_mem_solutions_df = pd.read_csv('optimal_mem_solutions.csv')

    # example LLM-Perf data entry
    # create a new LLM-Perf data entry as a single-row dataframe (example values)
    new_entry = pd.DataFrame({
        'Memory (MB)': [1500],
        'Backend': ['TensorFlow'],
        'Precision': ['float32'],
        'Quantization': ['None'],
        'Attention': ['Standard'],
        'Kernel': ['Default'],
        'Energy (tokens/kWh)': [5000],
        'Open LLM Score (%)': [80.0],
        'Params (B)': [1.2],
        'Experiment': ['Test'],
        'Prefill (s)': [0.5],
        'Decode (tokens/s)': [20.0],
        'End-to-End (s)': [2.5],
        'Architecture': ['Transformer']
    })

    # option to load the new LLM-Perf data entry, if preferred
    # load the new LLM-Perf data entry
    #new_entry_df = pd.read_csv('new_entry.csv')

    # get the recommendation for the new entry
    recommendation = recommend_optimized_observation(new_entry, optimal_mem_solutions_df)

    # save the recommendation to a CSV file
    recommendation.to_csv('recommendation.csv', index=False)

    print("Recommendation saved to recommendation.csv")