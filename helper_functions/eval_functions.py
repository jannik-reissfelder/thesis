import pandas as pd
import numpy as np


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


