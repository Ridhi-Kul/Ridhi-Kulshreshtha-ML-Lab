Machine Learning Lab: K-Means Clustering on Heart Disease Data
Project Overview

This lab focuses on applying K-Means Clustering to group patients based on medical attributes in a heart disease dataset.
The objective is to identify natural groupings (clusters) in the data and analyze patterns without using predefined labels.

Dataset

The dataset contains various medical features related to heart health, such as:

Age
Sex
Chest Pain Type
Blood Pressure
Cholesterol
Heart Rate
Other clinical attributes

Implementation Steps

1. Data Loading and Inspection
Loaded dataset using pandas
Checked structure using .head(), .info(), and .describe()

2. Data Preprocessing
Handled missing values (if any)
Encoded categorical variables
Applied feature scaling using preprocessing pipeline

3. Exploratory Data Analysis (EDA)
Analyzed feature distributions
Visualized relationships using plots
Identified patterns and variability in the dataset

4. Determining Optimal Number of Clusters
🔹 Elbow Method
Calculated inertia (WCSS) for different values of K
Observed a sharp decrease in inertia from K = 2 to K = 5
Selected approximate optimal K based on elbow point
🔹 Silhouette Score
Evaluated clustering quality for K = 2 to K = 9
Observed highest score at K = 2
Selected K = 4 considering balance between cluster quality and interpretability

5. K-Means Clustering
Applied K-Means with K = 4
Assigned cluster labels to each data point
Computed cluster centers

6. Dimensionality Reduction using PCA
Applied Principal Component Analysis (PCA)
Reduced data to 2 dimensions while retaining 90% variance
Enabled visualization of high-dimensional data

7. Cluster Visualization
Plotted clusters using PCA-transformed data
Observed clear separation between clusters
Identified compact and spread clusters

Observations
Clusters are reasonably well-separated
Some clusters are more compact, indicating strong similarity
Others are more spread, indicating variability
Silhouette scores indicate moderate cluster separation