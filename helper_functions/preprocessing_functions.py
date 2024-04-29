import  numpy as np
import pandas as pd

# Function to calculate the coefficient of variation
def calculate_cv(data):
    if np.mean(data) == 0:
        return np.nan  # Avoid division by zero
    return np.std(data, ddof=1) / np.mean(data)

def filter_targets(df_raw):
    df_quantile = df_raw.iloc[:, 80:].groupby(df_raw.index).quantile(0.85)
    candidates = df_quantile.drop(columns=[col for col in df_quantile.columns if df_quantile[col].eq(0).all()]).columns
    df_interim = df_raw[candidates]
    # Initialize an empty dictionary to store core microbiome data
    core_microbiome = {}

    for species in df_interim.index.unique():
        # Subset DataFrame by species, excluding the species column
        subset = df_interim.loc[[species]]

        # Calculate variability (CV) for each microorganism within this species
        variability = subset.apply(calculate_cv)

        # Filter based on a threshold, e.g., CV < 0.5 for low variability
        core_microbes = variability[variability < 0.5].index.tolist()

        # Store the core microorganisms in the dictionary
        core_microbiome[species] = core_microbes

        # Flatten the list of all microorganisms from the core microbiome of all species
        all_core_microorganisms = [microbe for microbes in core_microbiome.values() for microbe in microbes]

        # Convert the list to a set to remove duplicates, getting the unique set of core microorganisms
        unique_core_microorganisms = set(all_core_microorganisms)

        df_output = pd.concat([df_raw.iloc[:, :80], df_raw[list(unique_core_microorganisms)]], axis=1)

    return df_output, unique_core_microorganisms

