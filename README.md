# README for TrainerClass

## Module Description
The `trainer.py` module contains the `TrainerClass`, which is designed to facilitate the training and evaluation of various machine learning regression models. It supports different types of regression (single or multi-output), cross-validation methods (Leave-One-Out or K-Fold), and a selection of regression models. The class also includes methods for preprocessing data, selecting subsets of targets, initializing models, performing cross-validation, and visualizing results.

## Usage
- `regression_type`: Specifies whether to perform single or multi-output regression.
- `model_name`: Chooses the regression model to use, such as 'linear_regression', 'random_forest', 'xgboost', 'svm', 'mlp', 'ridge', or 'lasso'.
- `cv_type`: Determines the type of cross-validation, either 'loo' for Leave-One-Out or 'kfold' for K-Fold.
- `target_selection`: Chooses between 'abundant' or 'sparse' targets for regression.

### Example Use
To initialize the `TrainerClass` with all attributes set to true and import the class, you can use the following code snippet:

```python
from trainer import TrainerClass

# Initialize the TrainerClass with desired settings
trainer = TrainerClass(
    regression_type='multi',
    model_name='linear_regression',
    cv_type='kfold',
    target_selection='abundant'
)

# Example usage of the class
trainer.init_preprocess()
trainer.select_subset_of_targets_based_on_selected()
trainer.select_subset_of_targets_based_on_regression_type()
trainer.init_model_based_on_type()
trainer.cross_validation()
trainer.print_results()
trainer.run_visualization_if_needed()
```

This example demonstrates how to import the `TrainerClass` from `trainer.py`, initialize it with specific parameters, and use its methods to preprocess data, select targets, initialize the model, perform cross-validation, print results, and visualize the training and testing performance if the model is 'linear_regression'.