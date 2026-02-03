Machine Learning Lab 4: Linear Regression on US Housing Prices
Project Overview

This lab focuses on implementing Linear Regression to model and analyze housing prices using the USA Housing dataset.
The objective is to understand the relationship between housing price and multiple explanatory variables through exploratory data analysis, visualization, model training, and statistical interpretation of regression coefficients.

Dataset

The dataset used is USA_Housing.csv, which contains information about housing characteristics in different areas of the United States.

Features

Avg. Area Income

Avg. Area House Age

Avg. Area Number of Rooms

Avg. Area Number of Bedrooms

Area Population

Target Variable

Price – Price at which the house was sold

Implementation Steps
1. Data Loading and Inspection

Imported required libraries: NumPy, pandas, matplotlib, and seaborn.

Loaded the dataset using pandas.read_csv().

Inspected the dataset using .head(), .info(), and .describe() to understand structure, data types, and statistical properties.

Extracted column names using .columns.

2. Exploratory Data Analysis (EDA)

Visualized pairwise relationships between features using Seaborn pairplots.

Analyzed the distribution of the target variable (Price) using histogram and density plots.

Computed the correlation matrix to understand relationships between numerical features.

Visualized correlations using a heatmap.

3. Feature Selection

Created a list of all column names.

Selected numerical features as input variables (X).

Selected Price as the dependent variable (y).

Ignored the Address column as it is non-numeric.

4. Train–Test Split

Split the dataset into training and testing sets using train_test_split().

Used a test size of 30% and a fixed random seed for reproducibility.

Verified the shapes of training and testing datasets.

5. Model Training

Imported and instantiated the Linear Regression model from scikit-learn.

Trained the model using the training dataset.

Extracted and displayed the intercept and regression coefficients.

Stored coefficients in a DataFrame for interpretation.

6. Statistical Analysis of Coefficients

Calculated prediction errors on the training dataset.

Computed standard errors of the regression coefficients manually.

Calculated t-statistics for each coefficient.

Combined coefficients, standard errors, and t-statistics into a single DataFrame for analysis.