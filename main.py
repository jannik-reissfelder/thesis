### Main Script to run the entire pipeline

import pandas as pd
from sklearn.linear_model import LogisticRegression
from preprocess import PreprocessingClass
from trainer_predictor import TrainerClass
import numpy as np


## Do some preprocessing which is universal to all models
df = pd.read_parquet("./data/Seeds/60_seeds_CSS_merged.gz")


# Function to calculate the coefficient of variation
def calculate_cv(data):
    if np.mean(data) == 0:
        return np.nan  # Avoid division by zero
    return np.std(data, ddof=1) / np.mean(data)

def filter_targets(df_raw):
    df_quantile = df_raw.iloc[:, 68:].groupby(df_raw.index).quantile(0.85)
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

        df_output = pd.concat([df_raw.iloc[:, :68], df_raw[list(unique_core_microorganisms)]], axis=1)

    return df_output, unique_core_microorganisms





## get sample distribution
sample_distribution = df.index.value_counts()
## get mean sample distribution
mean_sample_distribution = sample_distribution.mean()
print("Mean sample distribution:", mean_sample_distribution)
print("Median sample distribution:", sample_distribution.median())

## set upsampling degree per sepcies
species_degree_mapping = {}
for species in sample_distribution.index:
    if sample_distribution[species] < int(mean_sample_distribution) + 1:
        degree = mean_sample_distribution / sample_distribution[species]
        # print(species, distribution[species], degree)
        # make degree an integer round to next integer
        degree = int(degree) + 1
        species_degree_mapping[species] = degree
# print("Species degree mapping:", species_degree_mapping)


df_red, microbes_left = filter_targets(df)
print("Number Microbes left:", len(microbes_left))
print("Number of sample plant species left:", df_red.index.nunique())


# initialize an empty dataframe to store the predictions
predictions_all_species = pd.DataFrame()

# set the algorithm to use
ALGO_NAME = "random_forest"
# set augmentation to use
AUGMENTATION = False
# set the path to save according to the augmentation
AUGMENTATION_PATH = "non-augmentation" if not AUGMENTATION else "augmentation"

# Iterate over the all subspecies as holdouts
# for each subspecies in subspecies we give it to the preprocessor
for subspecies in df_red.index.unique():
    # store the ordering of the columns for later use
    index_trues = df_red.iloc[:, 68:].loc[[subspecies]].columns
    # copy the dataframe
    df_input = df_red.copy()
    print("Hold out subspecies:", subspecies)
    preprocessor = PreprocessingClass(hold_out_species=subspecies, data=df_input, mapping=species_degree_mapping, use_augmentation=AUGMENTATION)
    preprocessor.run_all_methods()

    # get X_train and Y_train for training and prediction
    X = preprocessor.X_final
    Y = preprocessor.Y_candidates_final
    # get X_hold_out and Y_hold_out to make predictions
    X_hold_out = preprocessor.X_hold_out
    Y_hold_out = preprocessor.Y_candidates_hold_out
    # get closest species
    closest_species = preprocessor.closest


    # give to trainer class
    trainer = TrainerClass(x_matrix=X, y_abundance_matrix=Y, x_hold_out=X_hold_out, y_hold_out=Y_hold_out, algorithm=ALGO_NAME, closest_species=closest_species)
    trainer.run_train_predict_based_on_algorithm()
    # retrieve predictions and cv_results from trainer
    predictions = trainer.predictions
    cv_results = trainer.cv_results
    print("Predictions done for subspecies:", subspecies)
    print(predictions.shape)

    # sort the predictions
    predictions_sorted = predictions.reindex(index_trues)
    # rename the column
    predictions_sorted.columns = [f"predicted_{subspecies}"]

    # append to the predictions_all_species
    predictions_all_species = pd.concat([predictions_all_species, predictions_sorted], axis=1)

# save the predictions based on the algorithm name
predictions_all_species.to_csv(f"./predictions/{AUGMENTATION_PATH}/{ALGO_NAME}_predictions.csv")
print("Predictions saved")
print("Process done!")








