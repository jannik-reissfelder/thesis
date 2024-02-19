# handle imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV


class TrainerClass:
    def __init__(self,
                 x_matrix: pd.DataFrame,
                 y_abundance_matrix: pd.DataFrame,
                 x_hold_out: pd.DataFrame,
                 y_hold_out: pd.DataFrame,
                 algorithm: str,
                 closest_species: list,
                 n_neighbors: int = 3,

                 ):

        self.algorithm = algorithm
        self.X_input_matrix = x_matrix
        self.Y_target_abundance = y_abundance_matrix
        self.X_hold_out = x_hold_out
        self.Y_hold_out = y_hold_out
        self.closest_species = closest_species
        self.n_neighbors = n_neighbors # this is internally used by the KNN model
        self.model = None
        self.cv_results = None
        self.cv_mse_scores = None
        self.cv_rmse_scores = None
        self.predictions = None

        self.model_regression_dict = {

            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            'gpr': GaussianProcessRegressor(random_state=42),
        }

        # define parameter grid for each model
        self.model_param_grid_dict = {
            'linear_regression': {},
            'random_forest': {
                'estimator__max_depth': [5, 10],
                'estimator__max_features': ['sqrt', None],
                'estimator__min_samples_split': [2],
                'estimator__min_samples_leaf': [1]
            },
            'elastic_net': {
                'estimator__alpha': [0.5, 1, 2],
                'estimator__l1_ratio': [0.3, 0.5, 0.7]
            },
            'gpr': {
                'estimator__n_restarts_optimizer': [0, 1, 2, 3, 4, 5],
                'estimator__normalize_y': [True, False]
            },
        }

    def create_stratified_folds(self):
        """
        Create stratified folds for cross-validation based on species information in the index.
        """
        # Assuming the species information is stored in the index as the last level
        # Adjust this according to your index structure
        species = self.X_input_matrix.index.get_level_values(-1)
        unique_species = np.unique(species)
        species_to_int = {name: i for i, name in enumerate(unique_species)}
        stratify_labels = np.array([species_to_int[specie] for specie in species])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stratified_folds = list(skf.split(self.X_input_matrix, stratify_labels))

        return stratified_folds

    def initialize_model(self):
        """
        initialize model
        :return:
        """
        self.model = MultiOutputRegressor(self.model_regression_dict[self.algorithm])
        print("model in use: ", self.model)

    def cross_validation(self):
        """
        Perform cross-validation with grid search and stratified splits.
        """
        # Initialize model
        self.initialize_model()

        # Create stratified folds
        stratified_folds = self.create_stratified_folds()

        # GridSearchCV with custom cv parameter
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=self.model_param_grid_dict[self.algorithm],
                                   scoring='neg_mean_squared_error',
                                   cv=stratified_folds,
                                   verbose=2,
                                   n_jobs=-1)

        grid_search.fit(self.X_input_matrix, self.Y_target_abundance)

        # Best parameters and scores
        print("Best parameters found: ", grid_search.best_params_)
        self.model = grid_search.best_estimator_

        # Calculate and store MSE and RMSE scores
        self.cv_results = grid_search.cv_results_
        self.cv_mse_scores = -self.cv_results['mean_test_score']  # Scores are negative MSE
        self.cv_rmse_scores = np.sqrt(self.cv_mse_scores)

        # Report
        print("Cross-validation MSE scores: ", self.cv_mse_scores)
        print("Cross-validation RMSE scores: ", self.cv_rmse_scores)

    def fit_predict_best_model(self):
        """
        Fit the model with the best parameters found during grid search on the entire dataset,
        then predict and evaluate on unseen (hold-out) data.

        Parameters:
        - X_hold_out: The features of the hold-out set.
        - Y_hold_out: The true target values of the hold-out set.

        Returns:
        - predictions: The predicted values for the hold-out set.
        - mse: Mean squared error for the hold-out set.
        - rmse: Root mean squared error for the hold-out set.

        """

        # Assuming self.model is already the best estimator from GridSearchCV
        # Fit the model on the entire dataset (self.X_input_matrix, self.Y_target_abundance)
        self.model.fit(self.X_input_matrix, self.Y_target_abundance)

        # Predict on the hold-out set
        self.prediction_matrix = self.model.predict(self.X_hold_out)
        self.predictions = self.prediction_matrix[0]
        self.predictions = pd.DataFrame(self.predictions, index=self.Y_hold_out.columns)

        # Calculate MSE and RMSE for the hold-out set
        best_model_mse = mean_squared_error(self.Y_hold_out, self.prediction_matrix, multioutput='raw_values')
        best_model_rmse = np.sqrt(best_model_mse)

        # Optionally, print or return the evaluation metrics
        print("MSE on hold-out set:", best_model_mse)
        print("RMSE on hold-out set:", best_model_rmse)

    def run_knn_model(self):
        # from closest species get the first 3 species
        self.n_knn = self.closest_species[:self.n_neighbors]
        print(self.n_knn)
        # locate the rows of the closest species
        self.Y_knn = self.Y_target_abundance.loc[self.n_knn]
        # predict the mean abundance for each target variable based on the closest species
        mean_abundance_nn = self.Y_knn.groupby(self.Y_knn.index).mean()
        self.predictions = mean_abundance_nn.mean()



    def run_train_predict_based_on_algorithm(self):
        """
        Run the entire training and prediction process based on the chosen algorithm.
        """
        if self.algorithm != "knn":
            self.cross_validation()
            self.fit_predict_best_model()
        else:
            self.run_knn_model()
