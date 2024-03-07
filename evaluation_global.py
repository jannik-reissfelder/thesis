import pandas as pd
import os
from helper_functions.eval_functions import compute_JS_divergence, compute_BC_dissimilarity, calculate_wasserstein_with_normalization, calculate_bhattacharyya_with_normalization
import matplotlib.pyplot as plt

# Load dataframe for evaluation
# the dataframe is the orignal ground truth samples averaged over their in class samples
# sample with less than 2 samples are removed
df = pd.read_parquet("./evaluation/base_75_quantile.gz")



# get files from directory
# files_predictions = os.listdir("./predictions/non-augmentation")[:1]
sub = pd.read_csv("./predictions/augmentation/knn_predictions.csv", index_col=0)
sub = sub.T

# concatenate both dataframes to compare
df_with_prediction = pd.concat([df, sub])

# get all species
species_list = df.index.tolist()



# prepare dictionary to store the results
results = {}

# iterate over all species
for species in species_list:
    print(species)
    # get any species by index name
    class_true_vs_predictions = df_with_prediction[df_with_prediction.index.to_series().str.contains(species)]
    # compute the distance between the two samples using the JS divergence, wassterstein and bray-curtis
    js = compute_JS_divergence(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])
    wst = calculate_wasserstein_with_normalization(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])
    bc = compute_BC_dissimilarity(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])
    bhatt_c, bhatt_d = calculate_bhattacharyya_with_normalization(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])

    # store the results
    results[species] = [js, wst, bc, bhatt_c]

# transform the dictionary to a dataframe
results_df = pd.DataFrame(results, index=["JS", "WST", "BC", "Bhatt-Sim"]).T
results_df.to_csv("./evaluation/global/augmented/knn_global_metrics_quantile_base.csv")






def plot_metric_from_dfs(errors, metric_name):
    """
    Plots a specified metric from a DataFrame that contains different metrics from multiple CSV files,
    where the metric columns are named with a prefix followed by an underscore and the metric name.

    Parameters:
    errors (DataFrame): A DataFrame containing the metrics from multiple CSV files.
    metric_name (str): The name of the metric column to plot (e.g., 'JS').
    """
    # Filter columns that end with the specified metric name
    metric_columns = [col for col in errors.columns if col.endswith(f"_{metric_name}")]

    if not metric_columns:
        print(f"No columns found for the metric '{metric_name}'.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))

    for col in metric_columns:
        plt.plot(errors.index, errors[col], label=col)

    plt.title(f"Comparison of {metric_name} Metric Across Different DataFrames")
    plt.xlabel("Index")
    plt.ylabel(metric_name)
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()

# Directory containing the CSV files
directory = "./evaluation/global/augmented"

# Get only CSV files from the directory
files_predictions = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Initialize an empty DataFrame to hold all metrics
errors = pd.DataFrame()
# Load files and merge their specified metric into one DataFrame
for file in files_predictions:
    print("Loading file:", file)
    sub = pd.read_csv(os.path.join(directory, file), index_col=0)

    # Extract the filename without the extension to use as the column prefix
    filename_without_ext = os.path.splitext(file)[0]

    # Rename columns to include the filename as a prefix for distinction
    sub.columns = [filename_without_ext + '_' + col for col in sub.columns]

    # Concatenate horizontally
    errors = pd.concat([errors, sub], axis=1)

# plot_metric_from_dfs(errors, 'BC')