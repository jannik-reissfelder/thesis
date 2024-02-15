### Main Script to run the entire pipeline

import pandas as pd
from preprocess import PreprocessingClass


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
# print("Mean sample distribution:", mean_sample_distribution)

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
    preprocessor = PreprocessingClass(hold_out_species="Brassica rapa", data=df_red, mapping = species_degree_mapping, use_augmentation=True)
    preprocessor.run_all_methods()

    X = preprocessor.X_final
    Y = preprocessor.Y_candidates_final
    print(Y.index.value_counts())
    print(Y.index.nunique())


    
    # get results and put into tainer class

    # get results and put into evaluator class

    # from preprocessor retrieve non_candidates
    non_candidates = preprocessor.non_candidates
    # make dictionary and assign them all zeros
    non_candidates_abundance = {key: 0 for key in non_candidates}
    break
    # merge with predictions from candidate micros
    final_predictions = {**predictions, **non_candidates_abundance}
    print(len(final_predictions))










