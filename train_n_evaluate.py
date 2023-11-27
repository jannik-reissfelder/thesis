# handle imports
import seaborn as sns
s import PreprocessingClass
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split






class TrainerClass:
    def __init__(self,
                 regression_type: str = 'single',
                 model_name: str = 'linear_regression',
                 cv_type: str = 'kfold',
                 target_selection: str = 'abundant',
            ):

        self.regression_type = regression_type
        self.cv_type = cv_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_name = model_name
        self.model = None
        self.target_selection = target_selection

        if self.cv_type == 'loo':
            self.cv = LeaveOneOut()
        elif self.cv_type == 'kfold':
            self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            raise ValueError("Invalid type of cross-validation. Choose 'loo' or 'kfold'.")

        self.mse_scores = []
        self.rmse_scores = []
        self.mape_scores = []

        self.r2_scores = [] if self.cv_type == 'kfold' else None
        self.r2_scores_over_outputs = [] if self.cv_type == 'kfold' else None

        self.model_regression_dict= {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'xgboost': XGBRegressor(), #TODO check if this works
            'svm': SVR(), # TODO check if this works
            'mlp': MLPRegressor(), # TODO check if this works
            'ridge': Ridge(),
            'lasso': Lasso(),
        }

    def init_preprocess(self):
        """
        initialize preprocessing class
        :return:
        """
        self.preprocess = PreprocessingClass()
        self.preprocess.run_all_methods()
        self.X = self.preprocess.X_final
        self.y = self.preprocess.Y_prime_final

    def select_subset_of_targets_based_on_selected(self):
        """
        select subset of targets based on selected
        :return:
        """
        # assign most abundant bacteria for later use
        Y_abundant_a = self.y["a154c259d8b91cca550b123697f550ac"]
        Y_abundant_b = self.y["f56a4b040c4e1dc4fcea53b756ba99e0"]
        Y_abundant_c = self.y["1f2861e28c6cb119e9baffdae57b7cef"]

        # assign less abundant bacteria for later use
        Y_sparse_a = self.y["712b60c68f2ef91e050e3628c071c9e6"]
        Y_sparse_b = self.y["e93add934c58f00f0cb2c824d462ab9e"]
        Y_sparse_c = self.y["3db74264eb646ebc5542dd9d7b6c1e9b"]

        # For later binary regression
        self.Y_abundants = {
            "Microbiom_A": Y_abundant_a,
            "Microbiom_B": Y_abundant_b,
            "Microbiom_C": Y_abundant_c,
        }

        # transform to dataframe
        self.Y_abundants = pd.DataFrame(self.Y_abundants)

        # For later binary regression
        self.Y_sparse = {
            "Microbiom_A": Y_sparse_a,
            "Microbiom_B": Y_sparse_b,
            "Microbiom_C": Y_sparse_c,
        }

        # transform to dataframe
        self.Y_sparse = pd.DataFrame(self.Y_sparse)

        if self.target_selection == 'abundant':
            self.y_selected = self.Y_abundants
        elif self.target_selection == 'sparse':
            self.y_selected = self.Y_sparse
        else:
            raise ValueError("Invalid type of target selection. Choose 'abundant' or 'sparse'.")

    # write a function that further selects from y_selected based on regression type 'single' or 'multi'
    def select_subset_of_targets_based_on_regression_type(self):
        """
        select subset of targets based on regression type
        :return:
        """
        if self.regression_type == 'single':
            self.y_final = self.y_selected[self.y_selected.columns[0]]
        elif self.regression_type == 'multi':
            self.y_final = self.y_selected
        else:
            raise ValueError("Invalid type of regression. Choose 'single' or 'multi'.")


    def init_model_based_on_type(self):
        """
        initialize model based on type
        :return:
        """
        if self.regression_type == 'single':
            self.model = self.model_regression_dict[self.model_name]
            print("model in use: ", self.model)
        elif self.regression_type == 'multi':
            self.model = MultiOutputRegressor(self.model_regression_dict[self.model_name])
            print("model in use: ", self.model)
        else:
            raise ValueError("Invalid type of regression. Choose 'single' or 'multi'.")

    def cross_validation(self):
        """
        cross validation
        :return:

        """
        # extract values from dataframe
        self.X = self.X.values
        self.y_final = self.y_final.values

        # Loop over the folds
        i = 0
        for train_index, test_index in self.cv.split(self.X):
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y_final[train_index], self.y_final[test_index]
            i += 1
            print(f"""Fitting Fold {i}""")
            # Fit the model
            self.model.fit(self.X_train, self.y_train)

            # Make predictions
            self.y_pred = self.model.predict(self.X_test)

            # Calculate the mean squared error for the current fold
            mse = mean_squared_error(self.y_test, self.y_pred)
            self.mse_scores.append(mse)
            rmse = np.sqrt(mse)
            self.rmse_scores.append(rmse)

            # define helper fuction for MAPE
            def mean_absolute_percentage_error(y_true, y_pred):
                """
                Takes as input y_true and y_pred and calculates the MAPE error between both vectors. Note that y_true and y_pred are both ndarray numpy arrays.
                """
                # Avoid division by zero and handle the infinity
                y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)

                # Compute the absolute percentage error
                ape = np.abs((y_true - y_pred) / y_true)

                # Calculate the MAPE
                mape = np.mean(ape) * 100  # Multiply by 100 to get the percentage

                return mape, ape

            # Calculate the mean absolute percentage error for the current fold
            mape, ape = mean_absolute_percentage_error(self.y_test, self.y_pred)
            self.mape_scores.append(mape)


            # Calculate R2 score for the current fold if using k-fold
            if self.cv_type == 'kfold':
                # Calculate R2 score for each output separately
                r2_per_output = r2_score(self.y_test, self.y_pred, multioutput='raw_values')
                # Calculate the average R2 score across all outputs
                r2_average = np.mean(r2_per_output)
                # Append the average R2 score to the list
                self.r2_scores.append(r2_average)
                self.r2_scores_over_outputs.append(r2_per_output)


    def print_results(self):
        """
        print results
        :return:
        """
        print(f"""MSE: {np.mean(self.mse_scores)}""")
        print(f"""RMSE: {np.mean(self.rmse_scores)}""")
        print(f"""MAPE: {np.mean(self.mape_scores)}""")
        if self.cv_type == 'kfold':
            print(f"""R2: {np.mean(self.r2_scores)}""")
            print(f"""R2 per output: {self.r2_scores_over_outputs}""")

    def visualize_train_test_performance(self):
        """
        visualize train test performance
        :return:
        """

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_final, test_size=0.1, random_state=42)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Print metrics
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared (R2): {r2}")

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Values')
        plt.show()

        # Plot residuals
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.show()

        # Plot predicted values vs residuals
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Predicted Values vs Residuals')
        plt.show()

    # write a function that applies visualize_train_test_performance() only if self.model is 'linear_regression'
    def run_visualization_if_needed(self):
        """
        run visualization if needed
        :return:
        """
        if self.model_name == 'linear_regression':
            self.visualize_train_test_performance()
        else:
            pass


