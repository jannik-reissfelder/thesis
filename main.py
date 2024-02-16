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

# choose algorithm to use

# Iterate over the all subspecies as holdouts
# for each subspecies in subspecies we give it to the preprocessor
for subspecies in df_red.index.unique():
    print("Hold out subspecies:", subspecies)
    preprocessor = PreprocessingClass(hold_out_species=subspecies, data=df_red, mapping = species_degree_mapping, use_augmentation=False)
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
    trainer = TrainerClass(x_matrix=X, y_abundance_matrix=Y, x_hold_out=X_hold_out, y_hold_out=Y_hold_out, algorithm="linear_regression", closest_species=closest_species)
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
    print(type(predictions)) # numpy.narray
    print(predictions.shape)
    print(type(non_candidates_abundance_df)) # pandas dataframe
    print(non_candidates_abundance_df.shape)
    # transform predictions to dataframe
    predictions = pd.DataFrame(predictions, index=Y_hold_out.columns)
    predictions_all = pd.concat([predictions, non_candidates_abundance_df])
    break











