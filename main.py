# This is the main module of the program. It is responsible for the preprocessing of the data, the training
# and the evaluation of the model.

import os
# handle imports
from preprocessing import PreprocessingClass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score, fbeta_score
from sklearn.metrics import auc
import os
import pandas as pd

# initialize the class

class PreprocessingClass
    def __init__(self,
                 regression_type: str = 'single',
                 use_augmentation: bool = True,
                 use_normalized_data: bool = True,
                 use_pca: bool = True,
                 std_threshold: int = 1
                 ):

        self.Y_prime = None
        self.regression_type = regression_type
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

        if self.use_normalized_data == True:
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
        self.union_microbes = set.union(*self.filtered_species.values())
        print("Number of microbes after filtering:", len(self.union_microbes))
        # Reassign Y to Y_prime
        self.Y_prime = self.Y[list(self.union_microbes)]





