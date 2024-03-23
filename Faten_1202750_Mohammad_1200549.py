#Mohammad Abu Shams 1200549
#Faten Sultan 1202750
#Sec 4&2
#Dr. Yazan Abu Farha

import csv
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

TEST_SIZE = 0.3# 30% Testing data and 70% Training data.
K = 3# 3 Nearest Neighbors.
class NN:
    def __init__(self, trainingFeatures, trainingLabels):
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        """
                Given a list of features vectors of testing examples
                return the predicted class labels (list of either 0s or 1s)
                using the k nearest neighbors
        """
        predictions = []
        for test_instance in features:
            distances = []
            for train_instance, label in zip(self.trainingFeatures, self.trainingLabels):
                dist = np.linalg.norm(np.array(test_instance) - np.array(train_instance))
                distances.append((dist, label))
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:k]
            classes = [neighbor[1] for neighbor in neighbors]
            prediction = max(classes, key=classes.count)
            predictions.append(prediction)
        return predictions #The predicted class labels.
        raise NotImplementedError

def load_data(filename):
    """
        Load spam data from a CSV file `filename` and convert into a list of
        features vectors and a list of target labels. Return a tuple (features, labels).

        features vectors should be a list of lists, where each list contains the
        57 features vectors

        labels should be the corresponding list of labels, where each label
        is 1 if spam, and 0 otherwise.
    """
    features = []
    labels = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            features.append(list(map(float, row[:-1])))
            labels.append(int(row[-1]))
    return features, labels
    raise NotImplementedError

def preprocess(features):
    """
        normalize each feature by subtracting the mean value in each
        feature and dividing by the standard deviation
    """
    np_features = np.array(features)
    means = np.mean(np_features, axis=0)# The mean value.
    standard_dv = np.std(np_features, axis=0)# The standard deviation.
    return (np_features - means)/standard_dv
    raise NotImplementedError

def train_mlp_model(features, labels):
    """
        Given a list of features lists and a list of labels, return a
        fitted MLP model trained on the data using sklearn implementation.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=5000)
    mlp.fit(features, labels)
    return mlp
    raise NotImplementedError

def evaluate(labels, predictions):
    """
        Given a list of actual labels and a list of predicted labels,
        return (accuracy, precision, recall, f1).

        Assume each label is either a 1 (positive) or 0 (negative).
    """
    #Initial Value.
    tp = fp = tn = fn = 0
    for true, pred in zip(labels, predictions):
        if true == 1:
            if pred == 1:
                tp += 1# Increase true positive.
            else:
                fn += 1# Increase false negative.
        else:
            if pred == 1:
                fp += 1# Increase false positive.
            else:
                tn += 1# Increase True negative.

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * (precision * recall)) / (precision + recall)
    return accuracy, precision, recall, f1, tp, fp, tn, fn
    raise NotImplementedError

def main():
    filename = "spambase.csv"
    features, labels = load_data(filename)
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)

    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1, tp, fp, tn, fn = evaluate(y_test, predictions)

    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy1, precision1, recall1, f11, tp1, fp1, tn1, fn1 = evaluate(y_test, predictions)


    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~Nearest Neighbor Result~~~~~~~~~~~~~~~~~~~~~~")
    print("\t\t\t\t\t\tconfusion matrix ")
    print("                   Classified Positive                Classified Negative")
    print("Actual Positive         ", "TP=", tp, "                        ", "FN=", fn)
    print("Actual Negative         ", "FP=", fp, "                        ", "TN=", tn)



    print("**** ****")
    print("~~~~~~~~~~~~~~~~~~~~~MLP Results~~~~~~~~~~~~~~~~~~~~~~")
    print("\t\t\t\t\t\tconfusion matrix ")
    print("                   Classified Positive                Classified Negative")
    print("Actual Positive         ", "TP=", tp1, "                        ", "FN=", fn1)
    print("Actual Negative         ", "FP=", fp1, "                        ", "TN=", tn1)
    print("``````````````````````````````````````````````````````````````````````````````````")

    print("\n")
    print("\t\t\t\t -Nearest Neighbor Result- \t\t\t\t\t-MLP Result-  ")
    print("\n")

    print("Accuracy:        ", accuracy,"\t\t\t\t\t",accuracy1)
    print("Precision:       ", precision,"\t\t\t\t\t",precision1)
    print("Recall:          ", recall,"\t\t\t\t\t" ,recall1)
    print("F1:              ", f1,"\t\t\t\t\t" ,f11)



if __name__ == "__main__":
    main()
