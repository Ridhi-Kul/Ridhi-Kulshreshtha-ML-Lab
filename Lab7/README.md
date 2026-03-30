Machine Learning Lab: Breast Cancer Classification
Project Overview

This lab focuses on building a machine learning model to classify breast tumors as benign or malignant using the Breast Cancer Wisconsin (Original) dataset.
The workflow includes data preprocessing, exploratory data analysis, model training, and performance evaluation.

Dataset

The dataset used is the Breast Cancer Wisconsin (Original) dataset.

Problem Statement

Predict whether a tumor is:

Benign
Malignant

based on various cell characteristics.

Target Variable

Class

2 → Benign
4 → Malignant

Re-encoded as:

0 → Benign
1 → Malignant
Implementation Steps

1. Data Loading and Inspection
Loaded dataset using pandas.read_csv()
Assigned column names manually
Checked dataset structure using .info(), .shape(), and .describe()

2. Data Preprocessing
Replaced missing values (?) with null values
Removed rows containing missing values using dropna()
Converted data types where required
Re-encoded the target variable into binary format

3. Exploratory Data Analysis (EDA)
Performed univariate analysis using histograms and boxplots
Visualized feature relationships using correlation heatmap
Observed strong correlations between features and target variable

4. Feature Preparation
Removed unnecessary identifier column (Sample_code_number)
Separated dataset into:
X → Feature variables
y → Target variable

5. Train–Test Split
Split the dataset into training and testing sets
Ensured model evaluation on unseen data

6. Feature Scaling
Applied StandardScaler for Logistic Regression
Scaling was performed after train-test split
Target variable was not scaled

7. Random Forest Classification
Trained Random Forest model without scaling
Evaluated using:
Accuracy
Confusion Matrix
Classification Report

8. Hyperparameter Analysis

Analyzed the effect of different Random Forest parameters:

Number of Trees (n_estimators)
Criterion (Gini vs Entropy)
Max Depth
Minimum Samples Split

Plotted graphs to observe how these parameters affect accuracy.

9. Observations
Gini and Entropy show similar performance
Accuracy stabilizes after a certain number of trees
Limiting depth reduces overfitting but may lower accuracy
Increasing minimum samples split simplifies the model but may reduce flexibility