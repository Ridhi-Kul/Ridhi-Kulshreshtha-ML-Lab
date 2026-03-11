Machine Learning Lab: SVM and Decision Tree Classification

This lab focuses on implementing a Support Vector Machine and Decision Tree Classifier to perform classification on a dataset.
The objective is to preprocess the dataset, train model, and evaluate its performance using common classification metrics.

SVM

1. Overview

In addition to Decision Tree classification, a Support Vector Machine (SVM) model was implemented to perform classification on the dataset.
SVM is a supervised learning algorithm that finds the optimal decision boundary (hyperplane) that separates data points belonging to different classes.

2. Model Training

The Support Vector Classifier (SVC) from the scikit-learn library was used to train the SVM model.

The model learns the best hyperplane that maximizes the margin between different classes.

3. Hyperparameter Tuning

To improve the performance of the SVM model, Grid Search Cross Validation was applied to tune important hyperparameters.

The following parameters were explored:

C – Controls the trade-off between maximizing margin and minimizing classification error.

Gamma – Defines how far the influence of a single training example reaches.

Kernel – Determines the transformation applied to the data to make it linearly separable.

Example parameter grid used:

param_grid = {
'C': [0.1, 1, 10, 100, 1000],
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'kernel': ['rbf']
}

Grid Search evaluates different parameter combinations and selects the best-performing model.

P4. rediction

After training and tuning, the SVM model was used to predict the class labels of the test dataset.

5. Model Evaluation

The performance of the SVM model was evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

These metrics help assess the model’s ability to correctly classify unseen data.

DECISION TREE

1. Model Training

A Decision Tree Classifier was created using:

DecisionTreeClassifier()

The model was then trained using the training dataset.

2. Prediction

After training, the model was used to predict the class labels for the test dataset.

3. Model Evaluation

The performance of the model was evaluated using:

Accuracy Score

Confusion Matrix

Classification Report

These metrics help measure how well the model performs on unseen data.

4. Decision Tree Visualization

The trained decision tree was visualized using plot_tree() to understand how the model makes decisions based on feature values