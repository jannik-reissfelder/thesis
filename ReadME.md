# Predicting Microbial Community Composition from Plant Phylogeny

## Description
This repository contains the source code for my master's thesis project titled "Predicting the Microbial Community Composition from Plant Phylogeny". The goal of this project is to utilize plant phylogenetic information to predict microbial community compositions for 60 different plant species. This repository provides all necessary tools to preprocess data, train the predictive models, and evaluate their performance against the ground truth data.

## Installation
To set up this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/thesis-repo.git
cd thesis-repo
pip install -r requirements.txt
```

## Usage
To run the main pipeline that processes the data and trains the model, execute the main.py script:

```bash
Copy code
python main.py
```

This script initializes the preprocessing of the input data, followed by the training of models using the Preprocessor and TrainerClass.

To evaluate the models, run the evaluation.py script after the models have been trained:

```bash
Copy code
python evaluation.py
```
This will assess the model predictions using three main metrics and store the results per species.
## Additional Experimental Results

As detailed in my thesis under the section "Supplementary Material" (Appendix), comprehensive experimental results are provided for each of the 60 plant species analyzed. These results are critical for understanding individual performance metrics across different scenarios and algorithms used in the study.

The complete set of results is organized in the `evaluation` folder within this repository. Within this folder, you will find detailed results for three distinct scenarios:
- **Original**: Baseline results without any modifications.
- **Added Genetic Information**: Results incorporating additional genetic information into the analysis.
- **Data Augmentation**: Results utilizing data augmentation techniques to enhance model performance.

For each scenario, the results are further detailed on a per-plant-species basis, covering all algorithms and metrics used.

These details support the reproducibility of the research and allow for in-depth analysis of the models' performance as discussed in the thesis.

## Features
- **Data Preprocessing**: Standardizes and prepares plant phylogenetic data and microbial data for model training.
- **Model Training**: Trains models to predict microbial communities based on plant phylogeny. A suit of algorithms is available, including Random Forest, K-Nearest Neighbors, and Gaussian Process Regression.
- **Evaluation**: Compares predictions with actual microbial communities and quantifies model performance for three metrics: Bray-Curtis Distance, Jensen-Shannon Divergence, and Bhattacharyya Distance.




## Credits
This project was developed by Jannik Reißfelder. Special thanks to ATB Leibniz-Institute Potsdam, Department of Data Science, and all contributors to the project.

## Contact
For support or collaboration, please contact me.