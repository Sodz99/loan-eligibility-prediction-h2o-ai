# Loan Eligibility Prediction in Python using H2O.ai

You must implement a model that predicts if a loan should be granted to an individual based on the data provided

## Overview
When a customer applies for a loan at our company, we use statistical models to determine whether or not to grant the loan based on the likelihood of the loan being repaid. The factors involved in determining this likelihood are complex, and extensive statistical analysis and modelling are required to predict the outcome for each individual case.

## Data
The dataset used is an anonymized synthetic data that was generated specifically for use in this project. The data is designed to exhibit similar characteristics to genuine loan data. In this dataset, you must explore and cleanse a dataset consisting of over 1,00,000 loan records to determine the best way to predict whether a loan applicant should be granted a loan or not.

_Data is not included in this repository._

## Approach
1. EDA
- Missing data analysis
- Data imputation
- Bad features removal
- Features boxplot and distribution analysis basis visualization

2. Data cleaning / Pre-processing (outlier/missing values/categorical)
- One Hot encoder for categorical variables
- Label encoder if feature needs to be discretized

3. Feature Engineering
- Non-linear combination of existing features
- Add pre-calculated priors as features

4. Standardizing data
- Standard scaler

5. Modelling
- Random Forest
- Boosting Models
- Light GBM
- XGBoost
- GBT
- Neural Network & variants

6. Hyper parameter tuning
- GridSearchCV to find out the best model

7. Finalise Model - Model which has the best AUCPR on the labeled test data would be used. Although AUC and F1 score would also be tracked, the following metrics are also taken into account
- Precision/Recall
- AUC
- AUCPR
- F1 Score

8. Deployment
- Deployment using a Flask API and with a Docker container Note If you are using Windows OS, the XGBoost model might not work in your system since H2O-3.32.1.2 does not support XGBoost for windows. Please refer to this link for further details.

Modular code overview

Once you unzip the modular_code.zip file you can find the following folders within it.
1. input
2. src
3. output
4. lib
1. The input folder contains all the data that we have for analysis.
2. The src folder is the heart of the project. This folder contains all the modularized code for all the above steps in a modularized manner. It further contains the following. a. ML_pipeline b. deployment c. engine.py

The ML_pipeline is a folder that contains all the functions put into different python files which are appropriately named. These python functions are then called inside the engine.py file. The deployment folder contains all the files related to deploying the model

3. The output folder contains all the models that we trained for this data saved as reusable files. These models can be easily loaded and used for future use and the user need not have to train all the models from the beginning.
4. The lib folder is a reference folder. It contains the original ipython notebook that we saw in the videos.

## Tech stack
➔ Language : Python ➔ Libraries : Scikit-learn, H2O, pandas, numpy, flask, Seaborn, Matplotlib ➔ Containerization : Docker

## How to run
```bash
pip install -r modular_code/requirements.txt
python modular_code/src/engine.py
```
Notebooks: [`Multiple-Classification-h2o.ipynb`](modular_code/lib/Multiple-Classification-h2o.ipynb), [`Model_Api.ipynb`](modular_code/src/deployment/Model_Api.ipynb)

## Project structure
```
.gitignore
README.md
modular_code/
  lib/
    Multiple-Classification-h2o.ipynb
  requirements.txt
  src/
    ML_Pipeline/
    deployment/
    engine.py
```

## Credits
Based on a ProjectPro guided project.
