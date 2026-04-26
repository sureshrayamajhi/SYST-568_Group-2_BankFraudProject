# SYST-568_Group-2_BankFraudProject
Note:
-"Data Preprocessing with PCA, MARS, random forest and XGBoosting.R" is the final version with the fastest computational time and updated predictors.

-All the other files with the prefix "Initial" refers to the model trials with extremely high computational time and so the console is also attached as document file.

Project Overview

This project focuses on bank fraud detection using machine learning and statistical modeling techniques. The workflow includes data preprocessing, feature engineering, exploratory analysis, dimensionality reduction, and multiple predictive models (XGBoost, Random Forest, and MARS) to identify fraudulent transactions.

Dataset
Source: https://www.kaggle.com/datasets/orangelmendez/bank-fraud
The dataset (new_bank_fraud_detection.csv) contains transactional banking data including transaction time, amount, location, account details, and fraud labels. The target variable is Is_Fraud, which indicates whether a transaction is fraudulent or not.

Key engineered features include:

-Time-based features (Hour, Weekend, Night transactions)
-Behavioral ratios (Amount-to-Balance ratio, Log transformations)
-Location mismatch indicators
-Digital usage patterns
-Interaction features for time and amount

R Script Structure & Key Components

"Data Preprocessing with PCA, MARS, random forest and XGBoosting.R"

1. Data Loading & Libraries

-Loads required libraries (caret, dplyr, xgboost, randomForest, ggplot2, etc.) and imports the dataset.

2. Feature Engineering

Creates new predictive features such as:

-Transaction time breakdown (Hour, Night flag)
-Financial ratios (Amount vs Balance)
-Risk-based behavioral indicators
-Interaction terms for better fraud detection signals

3. Target Encoding

Encodes categorical variables (e.g., State, Branch, Location) using smoothed target encoding to capture fraud risk levels.

4. Data Cleaning & Encoding
Removes irrelevant identifiers (Transaction ID, Merchant ID, etc.)
Converts categorical variables to factors
Applies one-hot encoding for model readiness

5. Preprocessing

Uses caret preprocessing:

-Box-Cox transformation
-Centering and scaling
-Removal of near-zero variance features

6. Exploratory Analysis
-Correlation heatmap
-Skewness analysis
-Boxplots comparing fraud vs non-fraud distributions

7. Dimensionality Reduction (PCA)
-Identifies principal components explaining variance
-Determines components required for 80% variance coverage
-Analyzes top contributing features
-Visualizes PCA structure using scree plot and biplot

8. XGBoost Model
-Handles class imbalance using weighted parameters
-Trains gradient boosting model
-Evaluates using AUC and confusion matrix
-Extracts feature importance

9. Random Forest Model
-Uses downsampling to balance dataset
-Builds ensemble decision tree model
-Evaluates performance using AUC and confusion matrix
-Plots variable importance

10. MARS (Multivariate Adaptive Regression Splines)
-Builds flexible nonlinear classification model
-Tests multiple configurations:
-Strict pruning
-Penalization
-Cross-validation tuning
-Span control
-Compares performance using AUC, F1 score, and sensitivity
-Visualizes ROC curves and variable importance
