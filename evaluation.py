import pandas as pd
import os
from helper_functions.eval_functions import compute_distance_matrix, plot_heatmap
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


## Load dataframe for evaltuation
# the dataframe is the orignal ground truth samples averagged over their in class samples
# sample with less than 2 samples are removed
df = pd.read_parquet("./evaluation/base_avg.gz")

# get files from directory
files_predictions = os.listdir("./predictions/non-augmentation")


# now load files and merge them into one dataframe
predictions = pd.DataFrame()
for file in files_predictions[:2]:
    print("Loading file:", file)
    sub = pd.read_csv("./predictions/non-augmentation/" + file, index_col=0)
    predictions = pd.concat([predictions, sub], axis=1)
# transpose the dataframe
predictions = predictions.T

# choose class to evaluate
class_to_evaluate = "Brassica napus"

# get the predictions over the index that contains the class
# Convert the index to string and filter rows where the index contains "Brassica napus"
filtered_df = predictions[predictions.index.to_series().str.contains("Brassica napus")]

# concatenate the filtered dataframe with the original dataframe
df_with_prediction = pd.concat([df, filtered_df])

# Compute and plot  distance matrix based on metric
# metric = 'wasserstein'
# matrix = compute_distance_matrix(df_with_prediction, metric=metric)
# plot_heatmap(matrix, title='BC - Matrix', annotations=True)


### PCAAA #####
# Perform PCA
# num_components = 2
#
# pca = PCA(n_components=num_components)  # We'll look at the first two principal components
# pca_result = pca.fit_transform(df)  # First project original points
#
# # transform predictions into same space as above
# pca_result_pred = pca.transform(filtered_df)
#
# # concatenate two both pca results
# pca_result_all = np.concatenate([pca_result, pca_result_pred])
#
#
# # Create a DataFrame for the PCA results
# pca_df = pd.DataFrame(pca_result_all, columns=['PC1', 'PC2'])
#
# # Add the sample names as a column
# pca_df['sample'] = df_with_prediction.index # Adjust if necessary to match your data structure
# # pca_df['sample_detail'] = df.index
#
# # Calculate the percentage of variance explained by each component
# explained_var = pca.explained_variance_ratio_ * 100
#
# # Create an interactive scatter plot
# fig = px.scatter(pca_df, x='PC1', y='PC2', text = None, color = "sample",
#                  title=f'PCA of Microbiome Data (PC1: {explained_var[0]:.2f}%, PC2: {explained_var[1]:.2f}%)',
#                  color_discrete_sequence=px.colors.qualitative.Plotly)
#
# # Improve layout
# fig.update_traces(textposition='top center')
# fig.update_layout(height=600, width=1200)
#
# # Show the figure
# fig.show()


def plot_sorted_df_line_distribution(df_plot):
    plt.figure(figsize=(15, 10))  # Increased figure size for better clarity

    # Sort the DataFrame based on the values of the first row
    sorted_columns = df_plot.iloc[0].sort_values(ascending=False).index
    sorted_df = df_plot[sorted_columns].iloc[:, :100] # TODO: this is just for the first 100 columns

    for index, row in sorted_df.iterrows():
        # Plot each row of the sorted DataFrame
        plt.plot(row.values, label=f'Row {index}', linestyle='-', linewidth=1.5)

    plt.title('Line Graph of Each Row After Sorting Based on First Row')
    plt.xlabel('Column Index (Sorted)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


class_true_vs_predictions = df_with_prediction[df_with_prediction.index.to_series().str.contains("Brassica napus")]

plot_sorted_df_line_distribution(class_true_vs_predictions)


print("done") # debug point