# Machine Learning Lab 2: Census Income Data Preprocessing Project Overview

This lab focuses on data preprocessing and exploratory analysis using the Adult (Census Income) dataset.
The objective is to prepare the dataset for machine learning by handling missing values, detecting outliers, performing feature selection, and creating clean input features and target variables.

### Dataset

The dataset used is the Adult / Census Income dataset obtained from the UCI Machine Learning Repository.

Problem Statement

Predict whether an individual’s annual income exceeds $50K/year based on census data.

Target Variable

income

Implementation Steps
1. Data Loading

Loaded the dataset using the ucimlrepo library.

Combined feature set (X) and target (y) into a single DataFrame for analysis.

2. Missing Value Analysis

Identified missing values represented as '?'.

Replaced '?' with NaN values.

Visualized missing values using a heatmap.

Handled missing values using mode (categorical) and median (numerical) imputation.

3. Exploratory Data Analysis (EDA)

Identified numerical and categorical features.

Plotted correlation heatmap for numerical features.

Improved heatmap readability using increased figure size and annotation scaling.

4. Outlier Detection and Treatment

Detected outliers in numerical features using the IQR method.

Analyzed special cases such as capital-gain and capital-loss, which are zero-inflated.

Applied:

Log transformation for capital-related features.

Selective upper capping for hours-per-week.

Retained valid extreme values for other features.

5. Feature Selection

Computed correlation of numerical features with the target variable.

Dropped features with correlation outside the range ±0.2 to reduce noise.

6. Final Dataset Preparation

Separated the cleaned dataset into:

X: Input features

y: Target variable

The processed dataset is now ready for model training.