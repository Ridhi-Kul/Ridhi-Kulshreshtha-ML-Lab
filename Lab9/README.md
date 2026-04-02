Machine Learning Lab: Artificial Neural Network (ANN) for Breast Cancer Classification
Project Overview

This section of the lab focuses on implementing an Artificial Neural Network (ANN) to classify breast tumors as benign or malignant using preprocessed data from the Breast Cancer Wisconsin dataset.

The ANN model is trained on scaled features to learn complex patterns and improve classification performance.

ANN Implementation Steps
1. Data Preparation
Used preprocessed dataset from previous steps
Selected feature matrix (X) and target variable (y)
Performed train-test split
Applied StandardScaler to normalize input features

2. Model Architecture

A Sequential Neural Network was created with the following structure:

Input Layer: 8 features
Hidden Layer 1: 16 neurons (ReLU activation)
Hidden Layer 2: 8 neurons (ReLU activation)
Output Layer: 1 neuron (Sigmoid activation)

This architecture follows a decreasing pattern to progressively extract meaningful features.

3. Model Compilation

The model was compiled using:

Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy

This configuration is suitable for binary classification problems.

4. Model Training

The model was trained using:

Epochs: 20 (or increased for better performance)
Batch Size: 32
Validation Split: 20%

Training and validation accuracy were monitored to evaluate learning behavior.

5. Model Evaluation

The trained model was evaluated on the test dataset using:

Test Accuracy
Test Loss
Confusion Matrix

Predictions were obtained using:

y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)

6. Performance Visualization
Plotted training vs validation accuracy
Observed convergence and generalization
Identified minimal overfitting
Key Observations
The ANN model achieved high accuracy (~95–97%)
Training and validation curves showed stable learning
Feature scaling significantly improved performance
The model generalized well to unseen data
Conclusion

The ANN model effectively learned patterns from the dataset and achieved strong classification performance.
Compared to traditional machine learning models, ANN provides a flexible approach to capturing complex relationships in the data.

