import pandas as pd
import os
from helper_functions.eval_functions import compute_distance_matrix, plot_heatmap, perform_pca_and_visualize, plot_sorted_df_line_distribution, perform_pcoa_and_visualize
import numpy as np

## Load dataframe for evaltuation
# the dataframe is the original ground truth samples aggregated on species level
df = pd.read_parquet("../evaluation/groundtruth.gz")

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
# only get csv files
files = [file for file in files if file.endswith('.csv')]
# exclude "linear_regression" from the files
files = [file for file in files if "linear_regression" not in file]




# now load files and merge them into one dataframe
predictions = pd.DataFrame()
for file in files:
    print("Loading file:", file)
    sub = pd.read_csv(os.path.join(file_path, file), index_col=0)
    # rename columns to file name to keep track of the algorithms used
    # replace 'predicted_' + col_name to 'file_name' + col_name
    sub.columns = [col.replace('predicted_', file).replace('.csv', '_') for col in sub.columns]
    predictions = pd.concat([predictions, sub], axis=1)
# transpose the dataframe
predictions = predictions.T

# choose class to evaluate
class_to_evaluate = "Oryza alta"

# get the predictions over the index that contains the class
# Convert the index to string and filter rows where the index contains "Brassica napus"
predictions_eval_class = predictions[predictions.index.to_series().str.contains(class_to_evaluate)]

# add a column names "source" to predictions_eval_class
predictions_eval_class['source'] = 'predictions'

# concatenate the filtered dataframe with the original dataframe
df_core = df[predictions.columns]
# add a column names "source" to df_core
df_core['source'] = 'ground_truth'
df_with_prediction = pd.concat([df_core, predictions_eval_class])

# save the "source" column to a variable
source = df_with_prediction['source']
# drop the "source" column
df_with_prediction = df_with_prediction.drop(columns=['source'])

# Compute and plot  distance matrix based on metric
metric = 'bray-curtis'
matrix = compute_distance_matrix(df_with_prediction, metric=metric)
# plot_heatmap(matrix, title='BC - Matrix', annotations=True)



print("done") # debug point
# this plots the PCA of the original plant species vs 1 predicted plant species
# perform_pca_and_visualize(df_core, predictions_eval_class)
# this plots the PCoA of the original plant species vs 1 predicted plant species
perform_pcoa_and_visualize(matrix=matrix, mapping=source)



# this plots the abundance of the original plant species vs its predicted plant species
# class_true_vs_predictions = df_with_prediction[df_with_prediction.index.to_series().str.contains(class_to_evaluate)]
# plot_sorted_df_line_distribution(class_true_vs_predictions)

