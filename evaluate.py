import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix

sns.set()

def load(filepath):
    return load_model(filepath)

def get_results(model, data):
    return model.predict_generator(data)

def transform_binary_probabilities(results):
    probabilities = results.flatten()
    return probabilities

def transform_binary_predictions(results):
    predictions = 1 * (results.flatten() > 0.5)
    return predictions

def get_labels(data):
    return data.classes

def calculate_accuracy_loss(model, data):
    loss, accuracy = model.evaluate_generator(data)
    return loss, accuracy

def calculate_precision_recall_curve(data, results):
    """
    restricted to binary classifications
    returns precision, recall, thresholds
    """
    labels = get_labels(data)
    probabilities = transform_binary_probabilities(results)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return precision, recall

def calculate_average_precision(data, results):
    """
    restricted to binary classifications
    returns
    """
    labels = get_labels(data)
    probabilities = transform_binary_probabilities(results)
    average_precision = average_precision_score(labels, probabilities)
    return average_precision

def calculate_roc_curve(data, results):
    """
    restricted to binary classifications
    returns false positive rate, true positive rate
    """
    labels = get_labels(data)
    probabilities = transform_binary_probabilities(results)
    fpr, tpr , _ = roc_curve(labels, probabilities)
    return fpr, tpr

def calculate_confusion_matrix(data, results):
    """
    returns a confusion matrix
    """
    labels = get_labels(data)
    predictions = transform_binary_predictions(results)
    return confusion_matrix(labels, predictions)

def calculate_confusion_matrix_stats(data, results):
    confusion_matrix = calculate_confusion_matrix(data, results)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "AM": (TPR+TNR)/2,
        "GM": np.sqrt(TPR*TNR),
    }

def calculate_pr_auc(data, results):
    precision, recall = calculate_precision_recall_curve(data, results)
    return auc(recall, precision)

def plot_precision_recall(data, results):
    precision, recall = calculate_precision_recall_curve(data, results)
    plt.step(recall, precision)

def plot_roc_curve(data, results):
    fpr, tpr = calculate_roc_curve(data, results)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr)

def plot_confusion_matrix(data, results):
    confusion_matrix = calculate_confusion_matrix(data, results)
    labels = list(data.class_indices.keys())
    labels.sort()
    sns.heatmap(
            confusion_matrix,
            annot=True,
            cmap="YlGnBu",
            yticklabels=labels,
            xticklabels=labels,
            )
