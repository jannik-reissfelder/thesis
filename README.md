# README for PreprocessingClass

## Module Description
The `preprocessing.py` module contains the `PreprocessingClass`, which is responsible for the preprocessing of data in preparation for machine learning tasks. This class offers a variety of preprocessing functionalities, including data loading, normalization, variance filtering, target space reduction, data augmentation, and dimensionality reduction using PCA. The class is designed to handle different scenarios and can be customized with various options to suit the needs of the analysis.

## Usage
- `scenario`: Specifies the preprocessing scenario, such as 'baseline', 'manual', or 'optimization'.
- `use_augmentation`: Determines whether to apply Mixup data augmentation.
- `use_normalized_data`: Chooses the normalization method for microbiome data, such as 'TSS', 'CSS', 'DESeq', 'TMM', or 'absolute'.
- `use_pca`: Indicates whether to apply PCA for dimensionality reduction.
- `normalize_X`: Specifies the normalization method for the feature matrix X, such as 'False', 'minmax', or 'standard'.
- `std_threshold`: Sets the number of standard deviations to use for variance filtering.
- `target_limit`: Limits the number of target variables in the optimization scenario.

### Example Use
To initialize the `PreprocessingClass` with specific settings and import the class, you can use the following code snippet:

```python
from preprocessing import PreprocessingClass

# Initialize the PreprocessingClass with desired settings
preprocessor = PreprocessingClass(
    scenario='optimization',
    use_augmentation=True,
    use_normalized_data='CSS',
    use_pca=True,
    normalize_X='minmax',
    std_threshold=2,
    target_limit=50
)

# Example usage of the class
preprocessor.run_all_methods()
```

This example demonstrates how to import the `PreprocessingClass` from `preprocessing.py`, initialize it with specific parameters, and use its `run_all_methods` function to execute all preprocessing steps in the correct order. The steps include data loading, preprocessing, normalization, variance filtering, target space reduction, data augmentation, and dimensionality reduction with PCA.