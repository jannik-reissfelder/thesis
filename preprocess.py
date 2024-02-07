# This is the main module of the program. It is responsible for the preprocessing of the data, the training
# and the evaluation of the model.


# handle imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from gurobipy import Model, GRB, quicksum
import time

# initialize the class

class PreprocessingClass:
    def __init__(self,
                 data_source: str = "malus",
                 scenario: str = "baseline",
                 use_augmentation: bool = False,
                 use_normalized_data: str = "CSS",
                 use_pca: bool = False,
                 normalize_X: str = "False",
                 std_threshold: int = 1,
                 target_limit: int = 100,
                 hold_out_strategy =  "leave_one_species_out",
                 ):

        self.Y_prime = None
        self.use_augmentation = use_augmentation
        self.use_normalized_data = use_normalized_data
        self.use_pca = use_pca
        self.scenario = scenario
        self.input_path = None
        self.data = None
        self.X = None
        self.Y = None
        self.normalize_X = normalize_X
        self.std_threshold = std_threshold
        self.target_limit = target_limit
        self.hold_out_strategy = hold_out_strategy
        self.data_source = data_source

    def load_data(self, path: str = None):
        """
        This function loads the data from a given path and returns the data as a pandas dataframe.
        """
        self.data = pd.read_parquet("./data/Malus/data_merged.gz") if self.data_source == "malus" else pd.read_parquet("./data/Seeds/60_seeds_CSS_merged.gz")



    def set_hold_out_set(self):
        """
        This function sets the hold out set.

        """
        if self.hold_out_strategy == "leave_one_species_out":
            # Strategy 1: Leave 1 species out
            # removing the orientalis as a final hold out set
            # it is in a cluster with 4 closely other related species
            self.hold_out_set = self.data.loc["M. orientalis"] if self.data_source == "malus" else self.data.loc["Brassica rapa"] ## TODO adjust to correct seed species
            # removing the orientalis from further analysis
            self.data.drop(self.data.loc["M. orientalis"].index, inplace=True) if self.data_source == "malus" else self.data.drop(self.data.loc["Brassica rapa"].index, inplace=True)

        elif self.hold_out_strategy == "leave_one_out_per_species":
            # Strategy 2: Leave 1 out per species
            self.data.reset_index(inplace=True)
            self.hold_out_set = self.data.groupby("Species", group_keys=False).apply(lambda x: x.sample(1, random_state=42))
            # Important: I drop the hold out set from the data
            # otherwise it will be included in the training set
            # this might reduce variation for data augmentation
            self.data.drop(self.hold_out_set.index, inplace=True)
            # reset index to Species
            self.hold_out_set.set_index("Species", inplace=True)
            self.data.set_index("Species", inplace=True)
        elif self.hold_out_strategy == "leave_one_sample_out":
            # Strategy 3: Leave 1 sample out
            self.data.reset_index(inplace=True)
            self.hold_out_set =  self.data[self.data["Species"] == "M. orientalis"].sample(1, random_state=42)
            # Important: I drop the hold out set from the data
            # otherwise it will be included in the training set
            # this might reduce variation for data augmentation
            self.data.drop(self.hold_out_set.index, inplace=True)
            # reset index to Species
            self.hold_out_set.set_index("Species", inplace=True)
            self.data.set_index("Species", inplace=True)

        else:
            print("Invalid hold out strategy selected; raise ValueError.")
            raise ValueError("Invalid hold out strategy selected. Please choose 'leave_one_species_out' or 'leave_one_out_per_species'.")

    def preprocessing_malus(self, data: pd.DataFrame = None):
        """
        This function preprocesses the data.
        It assigns the target matrix and the feature matrix.
        """
        self.data.set_index('Species', inplace=True)
        self.data.drop(columns=["index"], inplace=True)

        # removing the "Pyrus communis" as a non-variant feature
        self.data.drop(columns="Pyrus communis", inplace=True)

        # applying the hold out strategy
        self.set_hold_out_set()

        # Splitting the dataframe into features and targets
        self.X = self.data.iloc[:, :11]
        self.Y = self.data.iloc[:, 11:]
        print("X-shape:", self.X.shape)
        print("Y-shape:", self.Y.shape)

        # do same for hold out set
        self.X_hold_out = self.hold_out_set.iloc[:, :11]
        self.Y_hold_out = self.hold_out_set.iloc[:, 11:]
        print("X_hold_out-shape:", self.X_hold_out.shape)
        print("Y_hold_out-shape:", self.Y_hold_out.shape)
    
    def preprocesses_seed_data(self, data: pd.DataFrame = None):
        """
        This function preprocesses the data.
        It assigns the target matrix and the feature matrix.
        """
        # self.data.set_index('Species', inplace=True)
        # self.data.drop(columns=["index"], inplace=True)

        # applying the hold out strategy
        self.set_hold_out_set()

        # Splitting the dataframe into features and targets
        self.X = self.data.iloc[:, :68]
        self.Y = self.data.iloc[:, 68:]
        print("X-shape:", self.X.shape)
        print("Y-shape:", self.Y.shape)

        # do same for hold out set
        self.X_hold_out = self.hold_out_set.iloc[:, :68]
        self.Y_hold_out = self.hold_out_set.iloc[:, 68:]
        print("X_hold_out-shape:", self.X_hold_out.shape)
        print("Y_hold_out-shape:", self.Y_hold_out.shape)
    
    def choose_preprocessing_method(self):
        """
        Chooses the preprocessing method based on the data source.
        """
        if self.data_source == "malus":
            self.preprocessing_malus()
        elif self.data_source == "seeds":
            self.preprocesses_seed_data()
        else:
            print("Invalid data source selected; raise ValueError.")
            raise ValueError("Invalid data source selected. Please choose 'malus' or 'seeds'.")

    
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
        valid_methods = ["TSS", "CSS", "DESeq", "TMM",  "absolute"]
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

    
    def normalize_X_(self):
        """
        This function normalizes the feature matrix X.
        """

        if self.normalize_X == "False":
            print("X normalization not applied; normalize_X is set to False.")
            self.X = self.X
        elif self.normalize_X == "minmax":
            print("X normalization applied; normalize_X is set to minmax.")
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        elif self.normalize_X == "standard":
            print("X normalization applied; normalize_X is set to standard.")
            self.X = (self.X - self.X.mean()) / self.X.std()
        else:
            print("Invalid normalization method selected; raise ValueError.")
            raise ValueError("Invalid normalization method selected. Please choose 'False', 'minmax', or 'standard'.")

    def filter_high_variance_outputs_manually(self):
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



    def filter_target_space_by_optimization(self):


        # Placeholder for the final set of selected target variables across all species
        self.selected_targets_union = set()


        # get unique species
        species_names = self.Y.index.unique()

        # Calculate the within-species variance for each column and store it in a dictionary
        self.within_species_variances = {}
        for j in self.Y.columns:
            self.within_species_variances[j] = self.Y.groupby('Species')[j].var().sum()

        # Calculate the variance of each target variable across species
        self.across_species_variances = self.Y.groupby("Species").mean().var()

        # Weight parameter to balance the two objectives
        self.gamma = 0.5 # This can be adjusted based on the relative importance of the objectives
        
        # Create a new Gurobi model for species 's'
        self.optimizer = Model(f"species_selection")
        
        # Add binary decision variables for each target variable 'j'
        decision_vars = self.optimizer.addVars(self.Y.columns, vtype=GRB.BINARY, name="select")
        
        # Add constraint to limit the number of selected target variables 
        self.optimizer.addConstr(quicksum(decision_vars[j] for j in self.Y.columns) == self.target_limit, "TargetLimit")
        
        # Now we will use the pre-calculated within variances from the dictionary
        self.sum_within_species_variances = quicksum(
            decision_vars[j] * self.within_species_variances[j] for j in self.Y.columns
        )

        # Set the combined objective function
        # Minimize within-species variance and maximize across-species variance
        self.optimizer.setObjective(
            quicksum(decision_vars[j] * (self.gamma * self.sum_within_species_variances[j] - (1 - self.gamma) * self.across_species_variances[j])
                    for j in self.Y.columns),
            GRB.MINIMIZE
        )
        
        
        # Solve the optimization problem
        self.optimizer.optimize()
        
        # Extract the selected target variables for species 's'
        selected_targets = {j for j, var in decision_vars.items() if var.X > 0.5}
        
        # Add the selected target variables to the union set
        self.selected_targets_union.update(selected_targets)
        

        print("Optimization complete.")
        print("Number of selected targets:", len(self.selected_targets_union))
        

        # Reassign Y to Y_prime
        print("Reassigning Y to Y_prime")
        self.Y_prime = self.Y[list(self.selected_targets_union)]
        
    def filter_outputs_based_on_scenario(self):
        """
        Filters output space based on the scenario.
        If "baseline" is selected, no filtering is applied.
        If "manual" is selected, method filter_high_variance_outputs_manually is applied.
        If "optimization" is selected, method filter_target_space_by_optimization is applied.
        """
        if self.scenario == "baseline":
            print("Baseline scenario selected; no filtering applied.")
            self.Y_prime = self.Y
        elif self.scenario == "manual":
            print("Manual scenario selected; filtering based on standard deviation applied.")
            time.sleep(3)
            self.filter_high_variance_outputs_manually()
        elif self.scenario == "optimization":
            print("Optimization scenario selected; filtering based on optimization applied.")
            time.sleep(3)
            self.filter_target_space_by_optimization()
        else:
            print("Invalid scenario selected; raise ValueError.")
            raise ValueError("Invalid scenario selected. Please choose 'baseline', 'manual', or 'optimization'.")
        



    def mixup_by_subspecies(self):
        '''Applies Mixup augmentation to the dataset within each subspecies.'''
        
        # Initialize empty DataFrames to hold the augmented data
        augmented_X = pd.DataFrame(columns=self.X.columns)
        augmented_Y = pd.DataFrame(columns=self.Y_prime.columns)

        # Get unique subspecies from the index of Y_prime
        subspecies = self.Y_prime.index.unique()

        # set mix_up_x False to only mix up the outputs
        self.mix_up_x = False

        def mixup_data(X, Y, alpha=1, mix_up_x=False, degree=20):
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

        # Iterate through each subspecies
        for species in subspecies:
            # Locate the rows for the current subspecies
            species_mask = self.Y_prime.index == species
            self.X_sub = self.X[species_mask]
            self.Y_sub = self.Y_prime[species_mask]

            # Apply mixup_data function to the current subspecies
            mixed_X_sub, mixed_Y_sub = mixup_data(self.X_sub, self.Y_sub)


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
        self.choose_preprocessing_method()
        self.normalize_microbiome_data()
        self.normalize_X_()
        self.filter_outputs_based_on_scenario()
        self.augment_data_if_needed()
        self.reduce_X_if_needed()

