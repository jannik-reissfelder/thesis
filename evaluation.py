import pandas as pd
import os
from helper_functions.eval_functions import compute_distance_matrix, plot_heatmap, perform_pca_and_visualize, plot_sorted_df_line_distribution, perform_pcoa_and_visualize
import numpy as np



## Load dataframe for evaltuation
# the dataframe is the orignal ground truth samples averagged over their in class samples
# sample with less than 2 samples are removed
df = pd.read_parquet("./evaluation/base_avg.gz")

# get files from directory
files_predictions = os.listdir("./predictions/non-augmentation")


# now load files and merge them into one dataframe
predictions = pd.DataFrame()
for file in files_predictions:
    print("Loading file:", file)
    sub = pd.read_csv("./predictions/non-augmentation/" + file, index_col=0)
    # rename columns to file name to keep track of the algorithms used
    # replace 'predicted_' + col_name to 'file_name' + col_name
    sub.columns = [col.replace('predicted_', file).replace('.csv', '_') for col in sub.columns]
    predictions = pd.concat([predictions, sub], axis=1)
# transpose the dataframe
predictions = predictions.T

# choose class to evaluate
class_to_evaluate = "Brassica napus"

# get the predictions over the index that contains the class
# Convert the index to string and filter rows where the index contains "Brassica napus"
predictions_eval_class = predictions[predictions.index.to_series().str.contains(class_to_evaluate)]

# concatenate the filtered dataframe with the original dataframe
df_with_prediction = pd.concat([df, predictions_eval_class])

# Compute and plot  distance matrix based on metric
metric = 'bray-curtis'
matrix = compute_distance_matrix(df_with_prediction, metric=metric)
plot_heatmap(matrix, title='BC - Matrix', annotations=True)

# this plots the PCA of the original plant species vs 1 predicted plant species
perform_pca_and_visualize(df, predictions_eval_class)
# this plots the PCoA of the original plant species vs 1 predicted plant species
perform_pcoa_and_visualize(matrix=matrix)



# this plots the abundance of the original plant species vs its predicted plant species
class_true_vs_predictions = df_with_prediction[df_with_prediction.index.to_series().str.contains(class_to_evaluate)]
plot_sorted_df_line_distribution(class_true_vs_predictions)


print("done") # debug point