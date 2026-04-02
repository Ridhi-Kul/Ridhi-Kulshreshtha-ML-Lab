Machine Learning Lab: Convolutional Neural Network (CNN) on MNIST Dataset
Project Overview

This lab focuses on implementing a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset.
The model learns spatial patterns such as edges and shapes using convolutional filters and achieves high accuracy in digit recognition.

Dataset

The MNIST dataset consists of grayscale images of handwritten digits:

Image size: 28 × 28 pixels
Total classes: 10 (digits 0–9)
Training samples: 60,000
Test samples: 10,000
Implementation Steps

1. Data Preprocessing

Reshaped data into CNN format:

(
𝑠
𝑎
𝑚
𝑝
𝑙
𝑒
𝑠
,
ℎ
𝑒
𝑖
𝑔
ℎ
𝑡
,
𝑤
𝑖
𝑑
𝑡
ℎ
,
𝑐
ℎ
𝑎
𝑛
𝑛
𝑒
𝑙
𝑠
)
=
(
60000
,
28
,
28
,
1
)
(samples,height,width,channels)=(60000,28,28,1)
Normalized pixel values from 0–255 → 0–1 for stable training

2. CNN Model Architecture

The model consists of the following layers:

Convolution Layer 1:
32 filters, kernel size (3×3), ReLU activation
Max Pooling Layer:
Pool size (2×2)
Convolution Layer 2:
64 filters, kernel size (3×3), ReLU activation
Max Pooling Layer:
Pool size (2×2)
Flatten Layer:
Converts 2D feature maps into 1D vector
Dense Layer:
128 neurons, ReLU activation
Dropout Layer:
50% dropout to prevent overfitting
Output Layer:
10 neurons, Softmax activation

3. Model Compilation
The model was compiled using:
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy
This setup is suitable for multi-class classification with integer labels.

4. Model Training
Trained using training dataset
Used multiple epochs for learning
Monitored training performance

5. Model Evaluation
Evaluated model on test dataset
Obtained:
Test Accuracy
Test Loss

6. Prediction
Generated predictions using:
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
Converted probability outputs into class labels

7. Confusion Matrix
Evaluated model performance across all 10 classes
Observed high accuracy with most predictions along the diagonal
Identified minor misclassifications between similar digits