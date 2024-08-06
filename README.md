# Project Title:
Towards Greener AI: An Energy-Performance Metric and Framework Recommendation System for Large Language Models

# Overview:
This project develops an energy-performance scoring and recommendation system for Large Language Models (LLMs). The system normalizes data, calculates an energy-performance score, performs group-wise optimization, and provides recommendations based on new LLM-Perf data entries.

# Use:
Access the scripts of this project sequentially in the terminal by calling "python 'script'.py". Ensure that all scripts are saved in the same working directory along with the appropriate data LLM-Perf .csv file. Data MUST be in .csv format and named 'llm_perf_data.csv' for system to run. 

# Data:
1. llm_perf_data.csv - HuggingFace's LLM-Perf A10-24GB-150W leaderboard, extracted on June 5, 2024
2. llm_perf_data_A10080GB275W.csv - HuggingFace's LLM-Perf A100-80GB-275W leaderboard, extracted on June 20, 2024


# Scripts to call sequentially:
1. script_normalize_and_score.py
This script calculates the energy-performance score based on normalized variable values and pre-determined weights

2. script_groupwise_optimization.py
This script creates memory quantiles and finds the optimal energy-performance scored model per quantile

3. script_recommendation_system.py
This script returns the optimal observation for a given new LLM-Perf data entry. An important note is that this version of this script features a hard-coded LLM-Perf "new" data entry in the boilerplate construct code for presentation purposes. In true implementation to the HuggingFace LLM-Perf backend, this script will receive dynamic new data inputs rather than this static sample.

# Author: 
Kate Poole 
