import seaborn as sns
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import json

from vis.visualization import visualize_cam, visualize_saliency, overlay
from vis.utils.utils import load_img, normalize, find_layer_idx
from keras.models import load_model, Model
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix, roc_auc_score
from sklearn import manifold
import pandas
from config import config
import math
import string
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve, accuracy_score, roc_auc_score, average_precision_score, precision_score, recall_score

sns.set()

from notebook_vars import *

def load(filepath):
    return load_model(filepath)


def get_results(model, data):
    return model.predict_generator(data, steps=math.ceil(data.n/config.BATCH_SIZE))


def clean_filename(filename):
    return "-".join(filename.split("-")[1:])

def accession_from_filename(filename):
    return filename.split("-")[1].replace(".jpeg", "")

def transform_binary_probabilities(results):
    probabilities = results.flatten()
    return probabilities


def transform_binary_predictions(results):
    predictions = 1 * (results.flatten() > 0.5)
    return predictions


def get_labels(data):
    return data.classes


def calculate_accuracy_loss(model, data):
    loss, accuracy = model.evaluate_generator(data, steps=math.ceil(data.n/config.BATCH_SIZE))
    return loss, accuracy


def calculate_precision_recall_curve(labels, results):
    """
    restricted to binary classifications
    returns precision, recall, thresholds
    """
    probabilities = transform_binary_probabilities(results)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return precision, recall


def calculate_average_precision(labels, results):
    """
    restricted to binary classifications
    returns
    """
    probabilities = transform_binary_probabilities(results)
    average_precision = average_precision_score(labels, probabilities)
    return average_precision


def calculate_roc_curve(labels, results):
    """
    restricted to binary classifications
    returns false positive rate, true positive rate
    """
    probabilities = transform_binary_probabilities(results)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return fpr, tpr


def calculate_confusion_matrix(labels, results):
    """
    returns a confusion matrix
    """
    predictions = transform_binary_predictions(results)
    return confusion_matrix(labels, predictions)

def adjusted_wald(p, n, z=1.96):
    p_adj = (n * p + (z**2)/2)/(n+z**2)
    n_adj = n + z**2
    span = z * math.sqrt(p_adj*(1-p_adj)/n_adj)
    return max(0, p_adj - span), min(p_adj + span, 1.0)

def calculate_confusion_matrix_stats(labels, results):
    confusion_matrix = calculate_confusion_matrix(labels, results)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    Acc = (TN + TP) / (TN + TP + FN + FP)
    return {
        "Acc": Acc,
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
        #"AM": (TPR + TNR) / 2,
        #"GM": np.sqrt(TPR * TNR),
    }


def calculate_pr_auc(labels, results):
    precision, recall = calculate_precision_recall_curve(labels, results)
    return auc(recall, precision)


def plot_precision_recall(labels, results, experts=[]):
    precision, recall = calculate_precision_recall_curve(labels, results)
    auc = roc_auc_score(labels, results)
    stats = calculate_confusion_matrix_stats(labels, results)
    points = [{
        "name": "model default",
        "precision": stats["PPV"][1],
        "recall": stats["TPR"][1],
    }]
    if len(experts) > 0:
        points = [
            *points, *[{
                "name": e["name"],
                "precision": e["PPV"][1],
                "recall": e["TPR"][1],
            } for e in experts]
        ]
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=pandas.DataFrame(points),
        x="recall",
        y="precision",
        hue="name",
        ax=ax)
    ax.step(recall, precision)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.04)
    ax.text(
        1,
        0,
        s="auc={:.2f}".format(auc),
        horizontalalignment='right',
        verticalalignment='bottom')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig


def plot_roc_curve(labels, results, experts=[], name="model"):
    probabilities = transform_binary_probabilities(results)
    auc = roc_auc_score(labels, probabilities)
    stats = calculate_confusion_matrix_stats(labels, results)
    points = [{
        "name": "model default",
        "FPR": stats["FPR"][1],
        "TPR": stats["TPR"][1],
    }]
    fig, ax = plt.subplots()
    if len(experts) > 0:
        points = [
            *points, *[{
                "name": e["name"],
                "FPR": e["FPR"][1],
                "TPR": e["TPR"][1],
            } for e in experts]
        ]
    sns.scatterplot(
        data=pandas.DataFrame(points), x="FPR", y="TPR", hue="name", ax=ax)
    fpr, tpr = calculate_roc_curve(labels, results)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='b', linestyle='--', alpha=0.5)
    ax.text(
        1,
        0,
        s="auc={:.2f}".format(auc),
        horizontalalignment='right',
        verticalalignment='bottom')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig


def plot_confusion_matrix(data, results):
    fig, ax = plt.subplots()
    confusion_matrix = calculate_confusion_matrix(get_labels(data), results)
    labels = list(data.class_indices.keys())
    labels.sort()
    sns.heatmap(
        confusion_matrix,
        annot=True,
        square=True,
        cmap="YlGnBu",
        cbar=False,
        yticklabels=labels,
        xticklabels=labels,
        ax=ax,
    )
    plt.xlabel('prediction', axes=ax)
    plt.ylabel('diagnosis by MRI or histopathology', axes=ax)
    return fig


def plot_tsne(model, layer_name, data, labels, fieldnames=None, perplexity=5):
    figures = list()
    intermediate_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict_generator(data, steps=math.ceil(data.n/config.BATCH_SIZE))
    embedding = manifold.TSNE(
        perplexity=perplexity).fit_transform(intermediate_output)
    for i, label in enumerate(labels):
        labelname = "label"
        if fieldnames is not None:
            labelname = fieldnames[i]
        pd = pandas.DataFrame.from_dict({
            "x": [d[0] for d in embedding],
            "y": [d[1] for d in embedding],
            labelname: label,
        })
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="x",
            y="y",
            data=pd,
            hue=labelname,
            hue_order=np.unique(label),
            ax=ax)
        ax.axis('off')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        figures.append(fig)
        plt.show()
    return figures


def get_expert_results(expert_file, files, expert_key="malignantBenign"):
    with open(expert_file) as o:
        expert = json.load(o)
        results = []
        for f in files:
            try:
                results.append(expert[clean_filename(f)][expert_key])
            except Exception as e:
                results.append(None)
                print("error with {}: {}".format(f, e))
        return results


def plot_expert_confusion(expert_file, dataset):
    results = np.array([
        dataset.class_indices.get(i, 0)
        for i in get_expert_results(expert_file, dataset.filenames)
    ])
    fig = plot_confusion_matrix(dataset, results)
    return calculate_confusion_matrix_stats(get_labels(dataset), results), fig


def plot_grad_cam(image_file,
                  model,
                  layer,
                  filter_idx=None,
                  backprop_modifier="relu"):
    image = load_img(
        image_file, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
    grad = visualize_cam(
        model,
        find_layer_idx(model, layer),
        filter_idx,
        normalize(image),
        backprop_modifier=backprop_modifier)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(overlay(grad, image))
    ax[0].axis('off')
    ax[1].imshow(image)
    ax[1].axis('off')
    return fig


def plot_multiple_grad_cam(
        images,
        model,
        layer,
        penultimate_layer=None,
        filter_idx=None,
        backprop_modifier=None,
        grad_modifier=None,
        experts=None,
        expert_spacing=0.1,
):
    rows = 2
    if experts is not None:
        rows = 3
    fig, ax = plt.subplots(
        rows, len(images), figsize=(4 * len(images), 4 * rows))
    ax = ax.flatten()
    penultimate_layer_idx = None
    if penultimate_layer:
        penultimate_layer_idx = find_layer_idx(model, penultimate_layer)
    for i, filename in enumerate(images):
        image = load_img(
            filename, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        ax[i].imshow(image)
        ax[i].axis('off')
    for i, filename in enumerate(images):
        image = load_img(
            filename, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        grad = visualize_cam(
            model,
            find_layer_idx(model, layer),
            filter_idx,
            normalize(image),
            penultimate_layer_idx=penultimate_layer_idx,
            backprop_modifier=backprop_modifier,
            grad_modifier=grad_modifier)
        ax[i + len(images)].imshow(overlay(grad, image))
        ax[i + len(images)].axis('off')
    if experts:
        for i, filename in enumerate(images):
            for j, expert in enumerate(experts):
                if i == 0:
                    message = "expert {}: {}".format(j + 1, expert[i])
                    ax[i + 2 * len(images)].text(
                        0.3,
                        1 - (expert_spacing * j),
                        message,
                        horizontalalignment='left',
                        verticalalignment='center')
                else:
                    message = "{}".format(expert[i])
                    ax[i + 2 * len(images)].text(
                        0.5,
                        1 - (expert_spacing * j),
                        message,
                        horizontalalignment='center',
                        verticalalignment='center')
            ax[i + 2 * len(images)].axis('off')
    return fig, ax


def plot_multiple_saliency(images,
                           model,
                           layer,
                           filter_idx=None,
                           backprop_modifier=None,
                           grad_modifier=None):
    fig, ax = plt.subplots(2, len(images), figsize=(4 * len(images), 4))
    ax = ax.flatten()
    for i, filename in enumerate(images):
        image = load_img(
            filename, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        ax[i].imshow(image)
        ax[i].axis('off')
    for i, filename in enumerate(images):
        grad = visualize_saliency(
            model,
            find_layer_idx(model, layer),
            filter_idx,
            normalize(image),
            backprop_modifier=backprop_modifier,
            grad_modifier=grad_modifier)
        image = load_img(
            filename, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        ax[i + len(images)].imshow(overlay(grad, image))
        ax[i + len(images)].axis('off')
    return fig


def get_pr_data_for_modality(dataset, comparison_models=[], modalities=MODALITIES): 
    results = list()
    points = list()
    for modality in modalities: 
        labels = dataset["{}-labels".format(modality)]
        probabilities = dataset["{}-probabilities".format(modality)]
        predictions = dataset["{}-predictions".format(modality)]
        print(modality, len(labels), len(probabilities), len(predictions))
        acc = accuracy_score(labels, predictions)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = auc(recall, precision)
        stats = calculate_confusion_matrix_stats(labels, probabilities)
        points.append({
            "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], pr_auc, acc),
            "precision": stats["PPV"][1],
            "recall": stats["TPR"][1],
        })
        for p, r in zip(precision, recall): 
            results.append({ "precision": p, "recall": r, "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], pr_auc, acc)})
    for probabilities in comparison_models: 
        modality = "Radiomics"
        labels = None
        for k in dataset.keys(): 
            if "labels" in k: 
                labels = dataset[k]
                break
        predictions = [p > 0.5 for p in probabilities]
        print(modality, len(labels), len(probabilities), len(predictions))
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = auc(recall, precision)
        stats = calculate_confusion_matrix_stats(labels, predictions)
        acc = accuracy_score(labels, predictions)
        points.append({
            "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], pr_auc, acc),
            "precision": stats["PPV"][1],
            "recall": stats["TPR"][1],
        })
        for p, r in zip(precision, recall): 
            results.append({ "precision": p, "recall": r, "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], pr_auc, acc)})           
    return results, pr_auc, []
        
def plot_multiple_precision_recall(dataset, experts=[], comparison_models=[], modalities=MODALITIES):
    results, auc, points = get_pr_data_for_modality(dataset, comparison_models, modalities=modalities)        
    if len(experts) > 0:
        for i, expert in enumerate(experts): 
            labels = None
            for k in dataset.keys(): 
                if "labels" in k: 
                    labels = dataset[k]
                    break
            predictions = expert
            stats = calculate_confusion_matrix_stats(labels, predictions)
            acc = accuracy_score(labels, predictions)
            points.append({
                "precision": stats["PPV"][1],
                "recall": stats["TPR"][1],                
                "experts": "Expert {} (acc={:.2f})".format(i + 1, acc), 
            })
    fig, ax = plt.subplots()
    seaborn.lineplot(
        data=pandas.DataFrame(results),
        x="recall",
        y="precision",
        hue="modality",
        ax=ax, 
        err_style=None,
    )
    if points: 
        seaborn.scatterplot(
            data=pandas.DataFrame(points),
            x="recall",
            y="precision",
            hue="experts",
            style="experts",                        
            markers=["o", "v", "s", "P"],
            palette={ p["experts"]: "black" for p in points },            
            ax=ax,
        )
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.02)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig

def get_roc_data_for_modality(dataset, comparison_models=[], modalities=MODALITIES): 
    results = list()
    points = list()
    for modality in modalities: 
        labels = dataset["{}-labels".format(modality)]
        probabilities = dataset["{}-probabilities".format(modality)]
        predictions = dataset["{}-predictions".format(modality)]
        fpr, tpr, _ = roc_curve(labels, probabilities, drop_intermediate=False)
        roc_auc = roc_auc_score(labels, probabilities)
        stats = calculate_confusion_matrix_stats(labels, probabilities)
        acc = accuracy_score(labels, predictions)
        points.append({
            "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], roc_auc, acc),
            "fpr": stats["FPR"][1],
            "tpr": stats["TPR"][1],
        })
        for f, t in zip(fpr, tpr): 
            results.append({ "fpr": f, "tpr": t, "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], roc_auc, acc)})
    for probabilities in comparison_models: 
        modality = "Radiomics"
        labels = None
        for k in dataset.keys(): 
            if "labels" in k: 
                labels = dataset[k]
                break
        predictions = [p > 0.5 for p in probabilities]
        fpr, tpr, _ = roc_curve(labels, probabilities, drop_intermediate=False)
        roc_auc = roc_auc_score(labels, probabilities)
        stats = calculate_confusion_matrix_stats(labels, predictions)
        acc = accuracy_score(labels, predictions)
        points.append({
            "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], roc_auc, acc),
            "fpr": stats["FPR"][1],
            "tpr": stats["TPR"][1],
        })
        for f, t in zip(fpr, tpr): 
            results.append({ "fpr": f, "tpr": t, "modality": "{} (auc={:.2f}, acc={:.2f})".format(MODALITY_KEY[modality], roc_auc, acc)})        
    return results, roc_auc, []

def get_reliability_data_for_modality(dataset, comparison_models=[], n_bins=10, modalities=MODALITIES): 
    results = list()
    points = list()
    for modality in modalities: 
        labels = dataset["{}-labels".format(modality)]
        probabilities = dataset["{}-probabilities".format(modality)]
        predictions = dataset["{}-predictions".format(modality)]
        positives, mean_predicted_value = calibration_curve(labels, probabilities, n_bins=n_bins)
        points.append({
            "modality": "{}".format(MODALITY_KEY[modality]),
            "positives": positives,
            "MPV": mean_predicted_value,
        })
        for p, m in zip(positives, mean_predicted_value): 
            results.append({ "positives": p, "mpv": m, "modality": "{}".format(MODALITY_KEY[modality])})
    for probabilities in comparison_models: 
        modality = "Radiomics"
        labels = None
        for k in dataset.keys(): 
            if "labels" in k: 
                labels = dataset[k]
                break
        positives, mean_predicted_value = calibration_curve(labels, probabilities, n_bins=n_bins)
        points.append({
            "modality": "{}".format(MODALITY_KEY[modality]),
            "positives": positives,
            "MPV": mean_predicted_value,
        })
        for p, m in zip(positives, mean_predicted_value): 
            results.append({ "positives": p, "mpv": m, "modality": "{}".format(MODALITY_KEY[modality])})        
    return results

def plot_multiple_reliability_curve(dataset, comparison_models=[], n_bins=10, modalities=MODALITIES):
    results = get_reliability_data_for_modality(dataset, comparison_models, n_bins, modalities=modalities)
    fig, ax = plt.subplots()
    seaborn.lineplot(
        data=pandas.DataFrame(results),
        x="mpv",
        y="positives",
        hue="modality",
        ax=ax,
        err_style=None,
    )
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.02)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig

def plot_multiple_roc_curve(dataset, experts=[], comparison_models=[], modalities=MODALITIES):
    results, auc, points = get_roc_data_for_modality(dataset, comparison_models, modalities=modalities)
    if len(experts) > 0:
        for i, expert in enumerate(experts): 
            labels = None
            for k in dataset.keys(): 
                if "labels" in k: 
                    labels = dataset[k]
                    break
            predictions = expert
            stats = calculate_confusion_matrix_stats(labels, predictions)
            acc = accuracy_score(labels, predictions)            
            points.append({
                "fpr": stats["FPR"][1],
                "tpr": stats["TPR"][1],
                "experts": "Expert {} (acc={:.2f})".format(i + 1, acc),                 
            })
    fig, ax = plt.subplots()
    seaborn.lineplot(
        data=pandas.DataFrame(results),
        x="fpr",
        y="tpr",
        hue="modality",
        ax=ax,
        err_style=None,
    )
    if points:     
        seaborn.scatterplot(
            data=pandas.DataFrame(points),
            x="fpr",
            y="tpr",
            hue="experts",
            style="experts",            
            ax=ax,
            markers=["o", "v", "s", "P"],
            palette={ p["experts"]: "black" for p in points },
        )
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-0.04, 1.02)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig

def get_statistics(dataset, experts=[], comparison_models=[], modalities=MODALITIES): 
    results = list()
    for modality in modalities: 
        labels = dataset["{}-labels".format(modality)]
        probabilities = dataset["{}-probabilities".format(modality)]
        predictions = dataset["{}-predictions".format(modality)]
        roc_auc = roc_auc_score(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = auc(recall, precision)
        f1 = f1_score(labels, predictions)
        d = {
            "F1 Score": [f1, f1],            
            "ROC AUC": [roc_auc, roc_auc], 
            "PR AUC": [pr_auc, pr_auc],
            **calculate_confusion_matrix_stats(labels, probabilities), 
            "Modality": [MODALITY_KEY[modality], MODALITY_KEY[modality]], 
            "Total": [len(labels), len(labels)], 
            "Malignant": [len([l for l in labels if l]), len([l for l in labels if l])], 
            "Benign": [len([l for l in labels if not l]), len([l for l in labels if not l])], 
        } 
        # remove more that are not relevant to imbalanced datasets
        del d["TP"]
        del d["TN"]
        del d["FN"]
        del d["FP"]    
        del d["FPR"]
        del d["FNR"]
        d["Acc (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["Acc"][1], *adjusted_wald(d["Acc"][1], len(labels)))]
        d["TPR (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["TPR"][1], *adjusted_wald(d["TPR"][1], len([l for l in labels if l])))]
        d["TNR (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["TNR"][1], *adjusted_wald(d["TNR"][1], len([l for l in labels if not l])))]
        d["acc-low"], d["acc-high"] = adjusted_wald(d["Acc"][1], len(labels))
        results.append(pandas.DataFrame(d).iloc[[1]])
    for i, expert in enumerate(experts): 
        for k in dataset.keys(): 
            if "labels" in k: 
                labels = dataset[k]
                break
        predictions = expert
        f1 = f1_score(labels, predictions)        
        d = {
            "F1 Score": [f1, f1],            
            **calculate_confusion_matrix_stats(labels, np.array(predictions)),             
            "Modality": ["Expert {}".format(i + 1), "Expert {}".format(i + 1)], 
            "Total": [len(labels), len(labels)], 
            "Malignant": [len([l for l in labels if l]), len([l for l in labels if l])], 
            "Benign": [len([l for l in labels if not l]), len([l for l in labels if not l])],             
        }
        # remove more that are not relevant to imbalanced datasets (remove from article too)
        del d["TP"]
        del d["TN"]
        del d["FN"]
        del d["FP"]               
        del d["FPR"]
        del d["FNR"]        
        d["Acc (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["Acc"][1], *adjusted_wald(d["Acc"][1], len(labels)))]
        d["TPR (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["TPR"][1], *adjusted_wald(d["TPR"][1], len([l for l in labels if l])))]
        d["TNR (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["TNR"][1], *adjusted_wald(d["TNR"][1], len([l for l in labels if not l])))]        
        d["acc-low"], d["acc-high"] = adjusted_wald(d["Acc"][1], len(labels))
        results.append(pandas.DataFrame(d).iloc[[1]])
    for probabilities in comparison_models: 
        labels = None
        for k in dataset.keys(): 
            if "labels" in k: 
                labels = dataset[k]
                break
        predictions = [p > 0.5 for p in probabilities]
        roc_auc = roc_auc_score(labels, probabilities)
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = auc(recall, precision)
        f1 = f1_score(labels, predictions)
        d = {
            "F1 Score": [f1, f1],            
            "ROC AUC": [roc_auc, roc_auc], 
            "PR AUC": [pr_auc, pr_auc],
            **calculate_confusion_matrix_stats(labels, probabilities), 
            "Modality": ["Radiomics", "Radiomics"], 
            "Total": [len(labels), len(labels)], 
            "Malignant": [len([l for l in labels if l]), len([l for l in labels if l])], 
            "Benign": [len([l for l in labels if not l]), len([l for l in labels if not l])],             
        } 
        # remove more that are not relevant to imbalanced datasets (remove from article too)
        del d["TP"]
        del d["TN"]
        del d["FN"]
        del d["FP"]         
        del d["FPR"]
        del d["FNR"]
        d["Acc (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["Acc"][1], *adjusted_wald(d["Acc"][1], len(labels)))]
        d["TPR (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["TPR"][1], *adjusted_wald(d["TPR"][1], len([l for l in labels if l])))]
        d["TNR (95% CI)"] = ["", "{:.2f} ({:.2f}-{:.2f})".format(d["TNR"][1], *adjusted_wald(d["TNR"][1], len([l for l in labels if not l])))]        
        d["acc-low"], d["acc-high"] = adjusted_wald(d["Acc"][1], len(labels))
        results.append(pandas.DataFrame(d).iloc[[1]])

    return pandas.concat(results, axis=0, sort=False).set_index("Modality")
