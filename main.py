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

# initialize the class

class MainClass:
    def __init__(self,
                 target_name: str = None,
                 use_pca: bool = True,
                 use_normalization: bool = True,
                 ):
    self.target_name = target_name
    self.use_pca = use_pca
    self.use_normalization = use_normalization
    # make a test comment
    # make a second test comment

    def load_data(self, path: str = None):
        """
        This function loads the data from a given path and returns the data as a pandas dataframe.
        """
        if path is None:
            path = os.getcwd()
        data = pd.read_csv(path)
        return data

    def preprocessing(self, data: pd.DataFrame = None):
        """
        This function preprocesses the data. It uses the PreprocessingClass from the preprocessing module.
        """
        if data is None:
            data = self.load_data()
        preprocessing = PreprocessingClass(data=data, target_name=self.target_name, use_pca=self.use_pca, use_normalization=self.use_normalization)
        preprocessing.preprocessing()
        return preprocessing

    def train_test_split(self, data: pd.DataFrame = None):

