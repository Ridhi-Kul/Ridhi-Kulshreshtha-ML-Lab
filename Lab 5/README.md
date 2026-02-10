Machine Learning Lab: Breast Cancer Classification Using Logistic Regression
Project Overview

This lab focuses on exploratory data analysis (EDA) and binary classification using the Breast Cancer Wisconsin (Original) dataset.
The objective is to preprocess the dataset, analyze feature relationships, and build a Logistic Regression model to predict whether a tumor is benign or malignant.

Dataset

The dataset used is the Breast Cancer Wisconsin (Original) dataset from the UCI Machine Learning Repository.

Problem Statement

Predict whether a breast tumor is:

Benign

Malignant

based on cell-related medical features.

Target Variable

Class

2 → Benign

4 → Malignant

Re-encoded as:

0 → Benign

1 → Malignant

Implementation Steps
1. Data Loading and Inspection

Loaded the dataset using pandas.read_csv() with assigned column names.

Checked dataset shape, column information, and statistical summary using .info() and .describe().

2. Missing Value Handling

Identified missing values in the Bare_nuclei column.

Since only a small number of rows contained missing values, they were removed using dropna().

3. Exploratory Data Analysis (EDA)

Performed univariate analysis using histograms and boxplots.

Computed correlation matrix and visualized it using a heatmap.

Observed that many features are strongly correlated with the target variable.

4. Feature Preparation

Removed unnecessary identifier columns (if present).

Converted the target variable into binary form (0 and 1).

Split the dataset into:

X: Input features

y: Target labels

5. Train–Test Split

Divided the dataset into training and testing sets using train_test_split() to evaluate model performance on unseen data.

6. Feature Scaling

Applied feature scaling using StandardScaler after the train–test split.

Scaling was performed only on input features, not on the target variable.

7. Logistic Regression Model Training

Trained a Logistic Regression classifier on the training dataset.

Generated predictions on the test dataset.

8. Model Evaluation

Evaluated performance using:

Accuracy

Confusion Matrix

Classification Report

Plotted the ROC Curve and calculated the AUC score.