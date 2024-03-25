import pandas as pd
import os
from helper_functions.eval_functions import compute_JS_divergence, compute_BC_dissimilarity, calculate_wasserstein_with_normalization, calculate_bhattacharyya_with_normalization
import matplotlib.pyplot as plt

# Load dataframe for evaluation
# the dataframe is the orignal ground truth samples averaged over their in class samples
# sample with less than 2 samples are removed
df = pd.read_parquet("./evaluation/groundtruth.gz")

# define path variables
# define augmentation scenario
AUGMENTATION = False
# set the path to save according to the augmentation
AUGMENTATION_PATH = "non-augmentation" if not AUGMENTATION else "augmentation"


####### FIRST PART #######
# set file path
file_path = f"./predictions/{AUGMENTATION_PATH}"
# get files from path
files = os.listdir(file_path)

for file in files:
    sub = pd.read_csv(os.path.join(file_path, file), index_col=0)
    # reduce df.T to the same columsn as sub
    df_core = df[sub.index].T
    # concatenate both dataframes to compare
    df_with_prediction = pd.concat([df_core, sub], axis=1).T

    # get all species
    species_list = df_core.columns.tolist()

    # prepare dictionary to store the results
    results = {}

    # iterate over all species
    for species in species_list:
        print(species)
        # get any species by index name
        class_true_vs_predictions = df_with_prediction[df_with_prediction.index.to_series().str.contains(species)]
        # compute the distance between the two samples using the JS divergence, wassterstein and bray-curtis
        jsd = compute_JS_divergence(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])
        wst = calculate_wasserstein_with_normalization(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])
        bc = compute_BC_dissimilarity(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])
        bhatt_c, bhatt_d = calculate_bhattacharyya_with_normalization(class_true_vs_predictions.iloc[0], class_true_vs_predictions.iloc[1])

        # store the results
        results[species] = [jsd, wst, bc, bhatt_d]

    # transform the dictionary to a dataframe
    results_df = pd.DataFrame(results, index=["JSD", "WST", "BC", "Bhatt"]).T
    print("debug")
    # save the results
    # remove csv suffix from file
    file = os.path.splitext(file)[0]
    # results_df.to_csv(f"./evaluation/global/{AUGMENTATION_PATH}/{file}_global_metrics.csv")




####### SECOND PART #######

def plot_metric_from_dfs(errors, metric_name, sort=False):
    """
    Plots a specified metric from a DataFrame that contains different metrics from multiple CSV files,
    where the metric columns are named with a prefix followed by an underscore and the metric name.
    Optionally sorts the observations in descending order before plotting.

    Parameters:
    errors (DataFrame): A DataFrame containing the metrics from multiple CSV files.
    metric_name (str): The name of the metric column to plot (e.g., 'JS').
    sort (bool): If True, sorts the observations in descending order before plotting. Default is False.
    """
    # Filter columns that end with the specified metric name
    metric_columns = [col for col in errors.columns if col.endswith(f"_{metric_name}")]

    if not metric_columns:
        print(f"No columns found for the metric '{metric_name}'.")
        return

    # Identify the column containing "random_forest" for sorting, if required and present
    rf_column = next((col for col in metric_columns if "random_forest" in col), None)

    if sort and rf_column:
        # Sort the DataFrame based on the "random_forest" metric column, in descending order
        errors = errors.sort_values(by=rf_column, ascending=False)

    # Plotting
    plt.figure(figsize=(15, 8))

    for col in metric_columns:
        plt.plot(errors.index, errors[col], label=col)

    plt.title(f"Comparison of {metric_name} Metric Across Different Hold-out Species")
    plt.xlabel("Index")
    plt.ylabel("Jensen-Shannon Divergence")
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to not cut off labels
    # plt.savefig(f"./evaluation/global/{AUGMENTATION_PATH}/{metric_name}_comparison.png")
    plt.show()

# set directory based on above setting
directory = f"./evaluation/global/{AUGMENTATION_PATH}"

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
    # also drop "global_metrics" from the filename
    filename_without_ext = filename_without_ext.replace("_global_metrics", "")
    # also drop "predictions" from the filename
    filename_without_ext = filename_without_ext.replace("_predictions", "")

    # Rename columns to include the filename as a prefix for distinction
    sub.columns = [filename_without_ext + '_' + col for col in sub.columns]

    # Concatenate horizontally
    errors = pd.concat([errors, sub], axis=1)
    print("Loaded and concatenated:", file)

plot_metric_from_dfs(errors, 'JSD', sort=False)

# write the mean of the errors to a csv file
errors_mean = errors.mean()
errors_mean.to_csv(f"./evaluation/global/{AUGMENTATION_PATH}/mean_errors_over_algorithms.csv")

