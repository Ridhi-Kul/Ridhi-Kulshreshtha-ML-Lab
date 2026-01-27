Machine Learning Lab: Breast Cancer Data Analysis
Project Overview

This lab focuses on exploratory data analysis (EDA) and data preprocessing using the Breast Cancer Wisconsin (Original) dataset.
The objective is to understand the dataset structure, handle missing values, analyze feature distributions, detect outliers, and prepare the data for machine learning.

Dataset

The analysis uses the Breast Cancer Wisconsin (Original) dataset.

Problem Statement

Predict whether a breast tumor is benign or malignant based on cell characteristics.

Features

The dataset contains the following attributes:

Clump_thickness

Uniformity_of_cell_size

Uniformity_of_cell_shape

Marginal_adhesion

Single_epithelial_cell_size

Bare_nuclei

Bland_chromatin

Normal_nucleoli

Mitoses

Target Variable

Class

2 → Benign

4 → Malignant
(Re-encoded to 0 → Benign, 1 → Malignant)

Implementation Steps
1. Data Loading and Inspection

Imported essential libraries: pandas, numpy, matplotlib, and seaborn.

Loaded the dataset and inspected it using .head(), .info(), and .describe().

Verified data types and identified missing values.

2. Missing Value Handling

Detected missing values in the Bare_nuclei column.

Since only 16 rows contained missing values, those rows were removed.

Verified that no missing values remained in the dataset.

3. Univariate Analysis

Analyzed feature distributions using histograms.

Used boxplots to visualize spread and detect outliers.

Computed unique value counts for each feature.

Observed that all features are ordinal categorical values in the range 1–10.

Visualized class distribution using a pie chart.

4. Outlier Detection

Applied the IQR method to detect outliers.

Identified that most flagged outliers were rare but valid medical scores.

Since features are ordinal and bounded, no outlier treatment was applied.

5. Target Variable Encoding

Re-encoded the target variable:

2 → 0 (Benign)

4 → 1 (Malignant)

6. Feature Scaling (Optional)

Observed that all features lie on the same 1–10 scale.

Min-Max scaling was optionally applied for compatibility with distance-based models.

The target variable was excluded from scaling.

7. Final Dataset Preparation

Separated the dataset into:

X: Input features

y: Target variable

The cleaned dataset is now ready for model training.