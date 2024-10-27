import pandas as pd
from scipy.stats import gmean  
from statistics import mean
import sys

def agg_relative_score(model_df: pd.DataFrame, baseline_df: pd.DataFrame):
    relative_score = model_df.drop("model", axis="columns") / baseline_df.drop(
        "model", axis="columns"
    )
    return relative_score.agg(gmean)

# Ensure that the paths to the CSV files are passed as arguments
if len(sys.argv) != 3:
    print("Usage: python script.py path/to/baseline/scores path/to/RAF/scores")
    sys.exit(1)

# Load the paths from the arguments
baseline_scores_path = sys.argv[1]
raf_scores_path = sys.argv[2]

# Read the CSV files
result_df = pd.read_csv(baseline_scores_path).set_index("dataset")
baseline_df = pd.read_csv(raf_scores_path).set_index("dataset")

# Calculate the aggregated relative score
agg_score_df = agg_relative_score(result_df, baseline_df)

# Print the result
print(agg_score_df)
