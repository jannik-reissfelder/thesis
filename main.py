### Main Script to run the entire pipeline

import pandas as pd
from preprocess import PreprocessingClass
from trainer_predictor import TrainerClass


## Do some preprocessing which is universal to all models
df = pd.read_parquet("./data/Seeds/60_seeds_CSS_merged.gz")

def remove_species_with_less_than_2_samples(data):
    species_to_keep = data.index.value_counts()[data.index.value_counts() >= 2].index
    return data.loc[species_to_keep]

df_red = remove_species_with_less_than_2_samples(df)
species_left = df_red.index.nunique()
# print("Number of species left:", species_left)


## get sample distribution
sample_distribution = df_red.index.value_counts()
## get mean sample distribution
mean_sample_distribution = sample_distribution.mean()
print("Mean sample distribution:", mean_sample_distribution)

## set upsampling degree per sepcies

species_degree_mapping = {}
for species in sample_distribution.index:
    if sample_distribution[species] < mean_sample_distribution:
        degree = mean_sample_distribution / sample_distribution[species]
        # print(species, distribution[species], degree)
        # make degree an integer round to next integer
        degree = int(degree) + 1
        species_degree_mapping[species] = degree
print("Species degree mapping:", species_degree_mapping)

# initialize an empty dataframe to store the predictions
predictions_all_species = pd.DataFrame()

# set the algorithm to use
ALGO_NAME = "knn"
# set augmentation to use
AUGMENTATION = True
# set the path to save according to the augmentation
AUGMENTATION_PATH = "non-augmentation" if not AUGMENTATION else "augmentation"

# Iterate over the all subspecies as holdouts
# for each subspecies in subspecies we give it to the preprocessor
for subspecies in df_red.index.unique():
    # store the ordering of the columns for later use
    index_trues = df_red.iloc[:, 68:].loc[subspecies].columns
    # copy the dataframe
    df_input = df_red.copy()
    print("Hold out subspecies:", subspecies)
    preprocessor = PreprocessingClass(hold_out_species=subspecies, data=df_input, mapping = species_degree_mapping, use_augmentation=False)
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
    trainer = TrainerClass(x_matrix=X, y_abundance_matrix=Y, x_hold_out=X_hold_out, y_hold_out=Y_hold_out, algorithm=ALGO_name, closest_species=closest_species)
    trainer.run_train_predict_based_on_algorithm()
    # retrieve predictions and cv_results from trainer
    predictions = trainer.predictions
    cv_results = trainer.cv_results
    print("Predictions done for subspecies:", subspecies)

    # from preprocessor retrieve non_candidates
    non_candidates = preprocessor.non_candidates
    # make dictionary and assign them all zeros
    non_candidates_abundance = {key: 0 for key in non_candidates}
    # transform to dataframe
    non_candidates_abundance_df = pd.DataFrame.from_dict(non_candidates_abundance, orient='index')
    # merge with predictions
    # transform predictions to dataframe
    predictions_all = pd.concat([predictions, non_candidates_abundance_df])
    # sort the predictions
    predictions_sorted = predictions_all.reindex(index_trues)
    # rename the column
    predictions_sorted.columns = [f"predicted_{subspecies}"]

    # append to the predictions_all_species
    predictions_all_species = pd.concat([predictions_all_species, predictions_sorted], axis=1)

# save the predictions based on the algorithm name
predictions_all_species.to_csv(f"./predictions/{ALGO_name}_predictions.csv")
print("Predictions saved")
print("Process done!")


def filter_targets(df_y):
    """
    This function filters the target space based on LogisticRegression.
    Its a mechanism of pre-fitlering and determining presence or absenceprior to regression
    """
    # make Y dataframe binary
    Y_binary = df_y.map(lambda x: 1 if x > 0 else 0)
    # iterate over targets and fit a logistic regression
    targets = Y_binary.columns
    # initialize an empty dataframe to store the predictions
    predictions_all_species = {}

    for i, target in enumerate(targets):
        print("Training model for target:", target, "Number:", i + 1, "out of", len(targets))
        model = LogisticRegression()

        # Fit the model on your training data
        model.fit(X, Y_binary[target])

        # Get predicted probabilities for the positive class (1)
        predicted_probabilities = model.predict_proba(X_hold_out)[:, 1]

        # Apply custom threshold to determine class labels
        custom_threshold = 0.5
        predictions_custom_threshold = (predicted_probabilities >= custom_threshold).astype(int)

        # store
        pred = predictions_custom_threshold[0]
        predictions_all_species[target] = pred

    # transform the predictions to a dataframe
    predictions_all_species = pd.DataFrame(predictions_all_species, index=[0])
    presence_df = predictions_all_species.T
    # rename column to "presence"
    presence_df.columns = ["presence"]
    present_micros = presence_df[presence_df["presence"] == 1].index
    return present_micros, len(present_micros)








