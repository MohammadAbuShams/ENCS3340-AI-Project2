# Spam Email Classification Project

This project aims to classify emails as spam or not spam using two machine learning algorithms: **K-Nearest Neighbors (K-NN)** and **Multi-Layer Perceptron (MLP)**. The dataset contains email features and labels indicating whether the email is spam.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Functions](#functions)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiments](#experiments)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Project Description

In this project, two machine learning algorithms are implemented and compared:
1. **K-Nearest Neighbors (K-NN)**: This algorithm assigns the class label to an instance based on the majority class of its k-nearest neighbors in the feature space. We use k = 3 in this project.
2. **Multi-Layer Perceptron (MLP)**: A neural network model with two hidden layers, trained using the scikit-learn library. The first hidden layer has 10 neurons, and the second has 5 neurons. The sigmoid activation function is used.

Both models are trained and evaluated on a dataset containing email features and labels.

## Dataset

The dataset, `spambase.csv`, contains 4601 examples. Each example has 58 attributes (57 features and 1 label). The label is `1` for spam and `0` for not spam.


### Functions

- **load_data(filename)**: Loads the dataset from the CSV file.
- **preprocess(features)**: Normalizes the features of the dataset using z-score normalization.
- **NN class**: Implements the K-Nearest Neighbors algorithm using Euclidean distance.
- **train_mlp_model()**: Trains the MLP model using scikit-learn's `MLPClassifier`.
- **evaluate()**: Computes evaluation metrics such as accuracy, precision, recall, and F1-score.


## Evaluation Metrics

The following metrics are used to evaluate the performance of the models:

- **Accuracy**: The proportion of correct predictions over the total number of instances.
- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positive instances.
- **F1-score**: The harmonic mean of precision and recall.

Formulae:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- Accuracy = (TP + TN) / (TP + FP + TN + FN)
- F1-score = 2 * (Precision * Recall) / (Precision + Recall)


## Experiments

- **K-NN**: Tested with different values of k. For this project, k = 3 was selected for final evaluation.
- **MLP**: Experimented with different architectures. The final network consists of two hidden layers with 10 and 5 neurons, respectively.

## Future Work

- **Improvements for MLP**: The MLP model could benefit from additional tuning of hyperparameters, such as increasing the number of layers or neurons, experimenting with different activation functions, or optimizing the learning rate.
- **Improvements for K-NN**: Explore other distance metrics like Manhattan or Minkowski distance and test with different values of k.
- **Feature Selection**: Applying feature selection techniques could reduce the dimensionality of the dataset, potentially improving model performance.

