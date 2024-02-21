import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from skbio.stats.ordination import pcoa


# Function to calculate Bray-Curtis dissimilarity between two samples
def compute_BC_dissimilarity(sample1, sample2):
    # Calculate the sum of the minimum values for each species present in both samples
    min_sum = np.sum(np.minimum(sample1, sample2))

    # Calculate the sum of all counts for each sample
    sum_sample1 = np.sum(sample1)
    sum_sample2 = np.sum(sample2)

    # Calculate the Bray-Curtis dissimilarity
    dissimilarity = 1 - (2 * min_sum) / (sum_sample1 + sum_sample2)

    return dissimilarity


from scipy.stats import wasserstein_distance


def compute_wasserstein(s_i, s_j):
    """
    Compute the Wasserstein distance between two samples.

    Parameters:
    s_i (np.array): The abundance profile of sample i.
    s_j (np.array): The abundance profile of sample j.

    Returns:
    float: The Wasserstein distance between the two samples.
    """
    return wasserstein_distance(s_i, s_j)


from scipy.spatial.distance import jensenshannon


def compute_JS_divergence(s_i, s_j):
    """
    Compute the Jensen-Shannon divergence between two samples.

    Parameters:
    s_i (np.array): The abundance profile of sample i.
    s_j (np.array): The abundance profile of sample j.

    Returns:
    float: The Jensen-Shannon divergence between the two samples.
    """
    return jensenshannon(s_i, s_j)


def compute_distance_matrix(df, metric='wasserstein'):
    """
    Compute the distance matrix for all pairs of samples in the dataframe using the specified metric.

    Parameters:
    df (pd.DataFrame): The dataframe containing the abundance profiles of all samples.
    metric (str): The metric to use for computing distances ('wasserstein', 'jensen-shannon', or 'bray-curtis').

    Returns:
    np.array: A matrix of distances.
    """
    num_samples = df.shape[0]
    distance_matrix = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            if metric == 'wasserstein':
                distance_matrix[i, j] = compute_wasserstein(df.iloc[i, :], df.iloc[j, :])
            elif metric == 'jensen-shannon':
                distance_matrix[i, j] = compute_JS_divergence(df.iloc[i, :], df.iloc[j, :])
            elif metric == 'bray-curtis':
                distance_matrix[i, j] = compute_BC_dissimilarity(df.iloc[i, :], df.iloc[j, :])
            else:
                raise ValueError("Invalid metric specified. Choose 'wasserstein', 'jensen-shannon', or 'bray-curtis'.")

    # Convert the distance matrix to a DataFrame for better readability
    distance_df = pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
    return distance_df


import seaborn as sns
import matplotlib.pyplot as plt


# Assuming 'distance_df' is the DataFrame containing the Bray-Curtis dissimilarities
# If you have the distance matrix as a numpy array, you can convert it to a DataFrame as shown previously

# Create a heatmap from the distance matrix
def plot_heatmap(distance_df, title='Distance Matrix Heatmap', annotations=True):
    # Set up the matplotlib figure
    plt.figure(figsize=(40, 40))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(distance_df, cmap='viridis', vmin=0, vmax=distance_df.max().max(), annot=annotations,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Adjust the plot
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(title)
    # plt.savefig('results/malus_data/evaluation/js_divergence.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()


def perform_pca_and_visualize(df, filtered_df):
    """
    Performs PCA on the given DataFrame and a filtered version of it, then visualizes the results.

    Parameters:
    - df: DataFrame, the original dataset.
    - filtered_df: DataFrame, a filtered version of the original dataset for prediction.

    Returns:
    - None, but displays a Plotly scatter plot.
    """
    num_components = 2

    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(df)  # Project original points
    pca_result_pred = pca.transform(filtered_df)  # Transform predictions into the same space
    pca_result_all = np.concatenate([pca_result, pca_result_pred])  # Concatenate results

    # Create a DataFrame for the PCA results
    pca_df = pd.DataFrame(pca_result_all, columns=['PC1', 'PC2'])
    pca_df['sample'] = df.index.tolist() + filtered_df.index.tolist()

    # Calculate the percentage of variance explained by each component
    explained_var = pca.explained_variance_ratio_ * 100

    # Create an interactive scatter plot
    fig = px.scatter(pca_df, x='PC1', y='PC2', text=None, color="sample",
                     title=f'PCA of Data (PC1: {explained_var[0]:.2f}%, PC2: {explained_var[1]:.2f}%)',
                     color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_traces(textposition='top center')
    fig.update_layout(height=600, width=1200)

    fig.show()



def plot_sorted_df_line_distribution(df_plot):
    plt.figure(figsize=(15, 10))  # Increased figure size for better clarity

    # Sort the DataFrame based on the values of the first row
    sorted_columns = df_plot.iloc[0].sort_values(ascending=False).index
    sorted_df = df_plot[sorted_columns].iloc[:, :300] # TODO: this is just for the first 200 columns

    for index, row in sorted_df.iterrows():
        # Plot each row of the sorted DataFrame
        plt.plot(row.values, label=f'Row {index}', linestyle='-', linewidth=1.5)

    plt.title('Line Graph of Each Row After Sorting Based on First Row')
    plt.xlabel('Column Index (Sorted)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()



def perform_pcoa_and_visualize(matrix):
    """
    Performs PCoA on the given distance matrix and visualizes the results.

    Parameters:
    - matrix: DataFrame, the distance matrix (e.g., Bray-Curtis distance matrix).
    - index_new: Index, the new index to be used for samples in the visualization.

    Returns:
    - None, but displays a Plotly scatter plot.
    """
    # Perform PCoA
    pcoa_results = pcoa(matrix)

    # Get the PCoA coordinates
    pcoa_coords = pcoa_results.samples[['PC1', 'PC2']]
    pcoa_coords.set_index(matrix.index, inplace=True)

    # Convert the index to a column for easy plotting
    pcoa_coords['sample_detail'] = pcoa_coords.index

    # Create an interactive scatter plot
    fig = px.scatter(pcoa_coords, x='PC1', y='PC2', text=None, color="sample_detail",
                     title='PCoA of Microbiome Data',
                     labels={
                         'PC1': f"PC1 - {pcoa_results.proportion_explained['PC1']*100:.2f}%",
                         'PC2': f"PC2 - {pcoa_results.proportion_explained['PC2']*100:.2f}%"
                     })

    # Optional: Adjust layout for better readability or aesthetics
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers+text'))
    fig.update_layout(height=600, width=1000, legend_title_text='Species')

    fig.show()

