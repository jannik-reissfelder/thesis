# This is the main module of the program. It is responsible for the preprocessing of the data, the training
# and the evaluation of the model.


# handle imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# initialize the class

class PreprocessingClass:
    def __init__(self,
                 use_augmentation: bool = True,
                 use_normalized_data: bool = True,
                 use_pca: bool = False,
                 std_threshold: int = 1
                 ):

        self.Y_prime = None
        self.use_augmentation = use_augmentation
        self.use_normalized_data = use_normalized_data
        self.use_pca = use_pca
        self.input_path = None
        self.data = None
        self.X = None
        self.Y = None
        self.std_threshold = std_threshold

    def load_data(self, path: str = None):
        """
        This function loads the data from a given path and returns the data as a pandas dataframe.
        """

        if self.use_normalized_data:
            self.data = pd.read_parquet("./data/data_merged.gz")
        else:
            self.data = pd.read_parquet("./data/data_merged_absolute.gz")

    def preprocessing(self, data: pd.DataFrame = None):
        """
        This function preprocesses the data.
        It assigns the target matrix and the feature matrix.
        """
        self.data.set_index('Species', inplace=True)
        self.data.drop(columns=["index"], inplace=True)
        # Splitting the dataframe into features and targets
        self.X = self.data.iloc[:, :12]
        self.Y = self.data.iloc[:, 12:]

        print("X-shape:", self.X.shape)
        print("Y-shape:", self.Y.shape)
    def filter_high_variance_outputs(self):
        """
        Filters out high variance output variables for each subspecies within the output matrix Y.

        Parameters:
        - num_std_dev: int or float, the number of standard deviations to use for the threshold.

        Returns:
        - filtered_outputs: dict, a dictionary with species names as keys and filtered DataFrames as values.
        """

        def filter_species_outputs(Y_species, num_std_dev):
            """Helper function to filter outputs for a single species."""

            # get mode across all output targets and remove those with mode greater than 0
            mode_filter = Y_species.mode(axis=0).iloc[0] > 0
            # get variance across all output targets and remove those with variance greater than threshold
            std_per_output = Y_species.std(axis=0)
            mean_std_across_output = std_per_output.mean()
            std_of_std_across_output = std_per_output.std()
            threshold = mean_std_across_output + num_std_dev * std_of_std_across_output
            variance_filter = std_per_output <= threshold
            # combine filters
            b = (variance_filter.values & mode_filter)
            print(len(set(Y_species.columns[b].values)))
            return set(Y_species.columns[b].values)

        self.filtered_outputs = {}
        species_names = self.Y.index.unique()
        for species_name in species_names:
            Y_species = self.Y.loc[species_name]
            print(species_name)
            Y_species_filtered_set = filter_species_outputs(Y_species, self.std_threshold)
            self.filtered_outputs[species_name] = Y_species_filtered_set

        # make union of all filtered outputs
        self.union_microbes = set.union(*self.filtered_outputs.values())
        print("Number of microbes after filtering:", len(self.union_microbes))
        # Reassign Y to Y_prime
        print("Reassigning Y to Y_prime")
        self.Y_prime = self.Y[list(self.union_microbes)]




    def mixup_by_subspecies(self):
        '''Applies Mixup augmentation to the dataset within each subspecies.'''
        # Initialize empty DataFrames to hold the augmented data
        augmented_X = pd.DataFrame(columns=self.X.columns)
        augmented_Y = pd.DataFrame(columns=self.Y_prime.columns)

        # Get unique subspecies from the index of Y_prime
        subspecies = self.Y_prime.index.unique()
        # set alpha for mixup
        self.alpha = 0.2
        # set mix_up_x False to only mix up the outputs
        self.mix_up_x = False

        def mixup_data(X, Y, alpha=0.2, mix_up_x=False):
            '''Helper Function to apply Mixup augmentation to the dataset.'''
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

            # If mix_up_x is True, also mix the inputs
            if mix_up_x:
                lam_x = lam.reshape(-1, 1)
                mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
                mixed_x_df = pd.DataFrame(mixed_x, columns=X.columns)
            else:
                # If mix_up_x is False, use the original inputs
                mixed_x_df = X

            # Convert mixed outputs to DataFrame
            mixed_y_df = pd.DataFrame(mixed_y, columns=Y.columns)

            return mixed_x_df, mixed_y_df

        # Iterate through each subspecies
        for species in subspecies:
            # Locate the rows for the current subspecies
            species_mask = self.Y_prime.index == species
            self.X_sub = self.X[species_mask]
            self.Y_sub = self.Y_prime[species_mask]

            # Apply mixup_data function to the current subspecies
            mixed_X_sub, mixed_Y_sub = mixup_data(self.X_sub, self.Y_sub)

            # Set the index of the mixed DataFrames to match the original subspecies index
            mixed_X_sub.index = self.Y_sub.index
            mixed_Y_sub.index = self.Y_sub.index

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
        self.Y_prime['source'] = 'original'


    def augment_data_if_needed(self):
        '''
        Executes mixup_by_subspecies if use_augmentation is True and
        concatenates the original data with the augmented data.
        If use_augmentation is False, reassings X_all and Y_prime_all to X and Y_prime.
        '''
        if self.use_augmentation:
            # Call the mixup_by_subspecies method
            self.mixup_by_subspecies()
            print("Data augmentation applied.")
            # concatenate the original data with the augmented data
            self.X_all = pd.concat([self.X, self.mixed_X])
            self.Y_prime_all = pd.concat([self.Y_prime, self.mixed_Y])

        else:
            print("Data augmentation not applied; use_augmentation is set to False.")
            # create a column called 'source' and set it to 'original'
            self.X['source'] = 'original'
            self.Y_prime['source'] = 'original'

            # reassing X_all and Y_prime_all to X and Y_prime
            self.X_all = self.X
            self.Y_prime_all = self.Y_prime


    def apply_pca(self):
        """
        Applies PCA to the feature matrix X.
        """
        # drop 'source' column
        self.X_all = self.X_all.drop(columns=['source'])
        # Applying PCA for dimensionality reduction
        self.num_components = 3
        pca = PCA(n_components=self.num_components)
        self.X_reduced = pca.fit_transform(self.X_all)

        # convert X_reduced to a DataFrame and set the column names
        self.X_reduced = pd.DataFrame(self.X_reduced)
        self.X_reduced.columns = ['PC1', 'PC2', 'PC3']

        # check how much variance is explained by the 50 components
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance by {self.num_components} components: {explained_variance.sum() * 100:.2f}%")

        # Visualizing the explained variance in a bar plot
        plt.bar(range(1, self.num_components + 1), explained_variance, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(1, self.num_components + 1), explained_variance.cumsum(), where='mid',
                 label='Cumulative explained variance')

        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.xticks(range(1, self.num_components + 1))
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def reduce_X_if_needed(self):
        """
        Executes apply_pca if use_pca is True and
        reassings X_all to X_reduced.
        If use_pca is False, reassings X_all to X.
        """
        if self.use_pca:
            # Call the apply_pca method
            self.apply_pca()
            print("PCA applied.")
            # reassign X_all to X_final
            self.X_final = self.X_reduced

        else:
            print("PCA not applied; use_pca is set to False.")
            # reassign X_all to X_final
            self.X_final = self.X_all.drop(columns=['source'])

        # reassign Y_prime_all to Y_prime_final
        self.Y_prime_final = self.Y_prime_all.drop(columns=['source'])

    def run_all_methods(self):
        """
        Runs all methods in the correct order.
        """
        self.load_data()
        self.preprocessing()
        self.filter_high_variance_outputs()
        self.augment_data_if_needed()
        self.reduce_X_if_needed()

