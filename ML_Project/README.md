❤️ Heart Disease Prediction using Machine Learning
📌 Project Overview

This project focuses on building a machine learning model to predict whether a person is likely to have heart disease based on medical attributes.
The workflow includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

📊 Dataset

The dataset contains various medical features such as:

Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol Level
Fasting Blood Sugar
Resting ECG
Maximum Heart Rate
Exercise-Induced Angina
Oldpeak
Slope
Number of Major Vessels
Thalassemia
🎯 Target Variable
Target (0 / 1)
0 → No Heart Disease
1 → Presence of Heart Disease


⚙️ Implementation Steps

1. Data Loading and Inspection
Loaded dataset using pandas
Explored data using .head(), .info(), .describe()

2. Data Preprocessing
Handled missing values (if any)
Encoded categorical variables
Performed feature scaling where required

3. Exploratory Data Analysis (EDA)
Visualized feature distributions using histograms
Used correlation heatmap to identify relationships
Analyzed patterns between features and target variable

4. Train-Test Split
Split dataset into training and testing sets
Ensured proper evaluation on unseen data

5. Model Training
Implemented multiple machine learning models such as 
Logistic Regression
Decision Tree Classifier
Support Vector Machine (SVM), etc.

6. Hyperparameter Tuning
Used GridSearchCV
For example in SVM optimization tuned parameters like:
C
gamma
kernel

7. Model Evaluation
Evaluated models using:

Accuracy Score
Confusion Matrix
Classification Report
ROC Curve and AUC Score
Precision-Recall graph

📦 Libraries Used
pandas
numpy
matplotlib
seaborn
scikit-learn

🚀 Results
Compared performance of multiple models
K-Nearest Neighbor performed best with an accuracy of 90.27% and Logistic Regression performed worst with 82.28% accuracy.