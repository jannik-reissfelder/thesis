# This is the main module of the program. It is responsible for the preprocessing of the data, the training
# and the evaluation of the model.


# handle imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time


# Additional helper functions

def find_closest_species(feature_vector, n_neighbors=5):
    """
    Finds the closest species based on phylogenetic distance.
    """

    closest_species = feature_vector.sort_values()[:n_neighbors].index.tolist()
    closest_species = pd.Series(closest_species).str.strip().tolist()
    return closest_species


def mixup_data(X, Y, alpha=1, mix_up_x=False, degree=1):
    '''Helper Function to apply Mixup augmentation to the dataset multiple times.'''
    # Initialize empty lists to store augmented data
    augmented_x_dfs = []
    augmented_y_dfs = []

    for _ in range(degree):
        # Convert X and Y to numpy arrays
        x = X.values
        y = Y.values
        if alpha > 0:
            # Sample λ from a Beta distribution
            lam = np.random.beta(alpha, alpha, x.shape[0])
        else:
            # No Mixup is applied if alpha is 0 or less
            lam = np.ones(x.shape[0])

        # Reshape λ to allow element-wise multiplication with x and y
        lam_y = lam.reshape(-1, 1)

        # Randomly shuffle the data
        index = np.random.permutation(x.shape[0])

        # Create mixed outputs
        mixed_y = lam_y * y + (1 - lam_y) * y[index, :]
        mixed_y_df = pd.DataFrame(mixed_y, columns=Y.columns)
        mixed_y_df.index = Y.index

        # If mix_up_x is True, also mix the inputs
        if mix_up_x:
            lam_x = lam.reshape(-1, 1)
            mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
            mixed_x_df = pd.DataFrame(mixed_x, columns=X.columns)
        else:
            # If mix_up_x is False, use the original inputs
            mixed_x_df = X

        # Append the mixed data to the lists
        augmented_x_dfs.append(mixed_x_df)
        augmented_y_dfs.append(mixed_y_df)

    # Concatenate augmented dataframes
    augmented_x_df_final = pd.concat(augmented_x_dfs, ignore_index=False)
    augmented_y_df_final = pd.concat(augmented_y_dfs, ignore_index=False)

    return augmented_x_df_final, augmented_y_df_final


# initialize the class

class PreprocessingClass:
    def __init__(self,
                 hold_out_species: str,
                 mapping: dict,
                 data: pd.DataFrame,
                 use_augmentation: bool = False,
                 use_normalized_data: str = "absolute",
                 candidate_n_neighbors: int = 6,
                 mix_up_x=False
                 ):

        self.hold_out_species = hold_out_species
        self.data = data
        self.mapping = mapping
        self.use_augmentation = use_augmentation
        self.use_normalized_data = use_normalized_data
        self.candidate_n_neighbors = candidate_n_neighbors
        self.mix_up_x = mix_up_x
        self.X = None
        self.Y = None
        self.Y_candidates = None
        self.closest = None

    def set_hold_out_set(self):
        """
        This function sets the hold out set.
        """
        # remove the hold out species from the distance matrix columns
        self.data.drop(columns=[self.hold_out_species], inplace=True)
        # removing the  hold out  species from the row data
        # first assign it to the hold_out_set
        self.hold_out_set = self.data.loc[self.hold_out_species]
        # removing the hold out species from the data
        self.data.drop(self.data.loc[self.hold_out_species].index, inplace=True)
        print("Hold out set shape:", self.hold_out_set.shape)
        print("Data shape after hold out:", self.data.shape)

    def preprocesses_seed_data(self, data: pd.DataFrame = None):
        """
        This function preprocesses the data.
        It assigns the target matrix and the feature matrix.
        """

        # applying the hold out strategy
        self.set_hold_out_set()

        # Splitting the dataframe into features and targets
        self.X = self.data.iloc[:, :67]
        self.Y = self.data.iloc[:, 67:]
        print("X-shape:", self.X.shape)
        print("Y-shape:", self.Y.shape)

        # do same for hold out set
        self.X_hold_out = self.hold_out_set.iloc[:, :67]
        self.Y_hold_out = self.hold_out_set.iloc[:, 67:]
        print("X_hold_out-shape:", self.X_hold_out.shape)
        print("Y_hold_out-shape:", self.Y_hold_out.shape)

    def get_candidate_microbes(self):
        """
        This function gets the candidate microbes according to the closest 
        neihbors of the hold out species.
        """
        # get all micros set
        total_micros = set(self.Y.columns)
        print("Total number of microbes found:", len(total_micros))
        hold_out_inst = self.X_hold_out.iloc[0]
        candidate_neighbors = hold_out_inst[self.X.index.unique()]
        # find closest species to hold out species
        self.closest = find_closest_species(candidate_neighbors, self.candidate_n_neighbors)
        print("closest species found: ", self.closest)
        candidate_rows = self.Y.loc[self.closest]
        # determine the candidate micros
        self.candidate_micros = candidate_rows.loc[:, (candidate_rows != 0).any(axis=0)].columns
        # assign the candidate micros to Y_candidates from Y
        self.Y_candidates = self.Y[self.candidate_micros]
        # also re-assign Y_hold_out to Y_candidates_hold_out
        self.Y_candidates_hold_out = self.Y_hold_out[self.candidate_micros]
        # get difference between total micros and candidate micros
        self.non_candidates = total_micros - set(self.candidate_micros)
        print("Number of candidate microbes found:", len(self.candidate_micros))
        print("Number of non-candidate microbes found:", len(self.non_candidates))
        print("Y_candidates-shape:", self.Y_candidates.shape)
        print("X-shape:", self.X.shape)

    def normalize_microbiome_data(self):
        """
        Normalize microbiome data using the specified method within Pandas DataFrame.

        Parameters:
        - data_frame: Pandas DataFrame
            Microbiome data matrix where samples are rows and OTUs are columns.
        - method: str
            Normalization method ("TSS", "CSS", "DESeq", "TMM", "GMPR", "absolute").

        Returns:
        - normalized_data_frame: Pandas DataFrame
            Normalized microbiome data matrix.
        """

        # Check if the method is valid
        valid_methods = ["TSS", "CSS", "DESeq", "TMM", "absolute"]
        if self.use_normalized_data not in valid_methods:
            raise ValueError("Invalid normalization method. Please choose from: {}".format(", ".join(valid_methods)))

        # Absolute: No normalization
        if self.use_normalized_data == "absolute":
            self.Y = self.Y

        # TSS Normalization
        elif self.use_normalized_data == "TSS":
            scale_factor = self.Y.sum(axis=1)
            self.Y = self.Y.div(scale_factor, axis=0)

        # CSS Normalization
        elif self.use_normalized_data == "CSS":
            scale_factor = self.Y.apply(lambda x: x.sum() / np.median(x[x > 0]), axis=0)
            self.Y = self.Y.div(scale_factor, axis=1)

        # DESeq Normalization
        elif self.use_normalized_data == "DESeq":
            scale_factor = self.Y.apply(lambda x: x.sum() / np.exp(np.mean(np.log(x[x > 0]))), axis=0)
            self.Y = self.Y.div(scale_factor, axis=1)

    def mixup_by_subspecies(self):
        '''Applies Mixup augmentation to the dataset within each subspecies.'''

        # Initialize empty DataFrames to hold the augmented data
        augmented_X = pd.DataFrame(columns=self.X.columns)
        augmented_Y = pd.DataFrame(columns=self.Y_candidates.columns)

        # Get unique subspecies from the index of Y_candidates
        subspecies = self.Y_candidates.index.unique()

        # Iterate through each subspecies (without the holdout species)
        for species in subspecies:
            # check whether the species is in the mapping and if not continue to the next species
            # reason: if species is not in mapping no need to augment
            if species not in self.mapping:
                continue

            species_specific_degree = self.mapping[species]

            # Locate the rows for the current subspecies
            species_mask = self.Y_candidates.index == species
            self.X_sub = self.X[species_mask]
            self.Y_sub = self.Y_candidates[species_mask]

            # Apply mixup_data function to the current subspecies
            mixed_X_sub, mixed_Y_sub = mixup_data(self.X_sub, self.Y_sub, mix_up_x=self.mix_up_x,
                                                  degree=species_specific_degree)

            # Append the mixed data to the augmented DataFrames
            augmented_X = pd.concat([augmented_X, mixed_X_sub])
            augmented_Y = pd.concat([augmented_Y, mixed_Y_sub])

        # set X and Y to augmented data
        self.mixed_X = augmented_X
        self.mixed_Y = augmented_Y

        # create a column called 'source' and set it to 'artificial'
        self.mixed_X['source'] = 'artificial'
        self.mixed_Y['source'] = 'artificial'

        # create a column called 'source' and set it to 'original'
        self.X['source'] = 'original'
        self.Y_candidates['source'] = 'original'

    def augment_data_if_needed(self):
        '''
        Executes mixup_by_subspecies if use_augmentation is True and
        concatenates the original data with the augmented data. Reassigns them to X_final and Y_candidates_final.
        If use_augmentation is False,  X and Y_candidates are copied and reassigned to X_final and Y_candidates_final.
        '''
        if self.use_augmentation:
            # Call the mixup_by_subspecies method
            self.mixup_by_subspecies()
            print("Data augmentation applied.")
            # concatenate the original data with the augmented data
            self.X_final = pd.concat([self.X, self.mixed_X])
            self.Y_candidates_final = pd.concat([self.Y_candidates, self.mixed_Y])
            # drop the source columns for both
            self.X_final.drop(columns=['source'], inplace=True)
            self.Y_candidates_final.drop(columns=['source'], inplace=True)
            # print final shapes
            print("X_final-shape:", self.X_final.shape)
            print("Y_candidates_final-shape:", self.Y_candidates_final.shape)

        else:
            print("Data augmentation not applied; use_augmentation is set to False.")
            # reassing X_final and Y_candidates_final to X and Y_candidates
            self.X_final = self.X
            self.Y_candidates_final = self.Y_candidates

    def run_all_methods(self):
        """
        Runs all methods in the correct order.
        """

        self.preprocesses_seed_data()
        self.get_candidate_microbes()
        self.normalize_microbiome_data()
        self.augment_data_if_needed()
