import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def load_data(file_path):
    return pd.read_csv(file_path)

def find_best_threshold_youden(df):
    thresholds = np.linspace(0, 1, 100)
    youdens_index = []
    for threshold in thresholds:
        predictions = (df['cosine_similarity'] > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['unionable'], predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        youden_index = sensitivity + specificity - 1
        youdens_index.append(youden_index)
    best_threshold = thresholds[np.argmax(youdens_index)]
    best_youden_index = max(youdens_index)
    return best_threshold, best_youden_index


def plot_roc_curve(df):
    fpr, tpr, thresholds = roc_curve(df['unionable'], df['cosine_similarity'])
    auc = roc_auc_score(df['unionable'], df['cosine_similarity'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    return fpr, tpr, thresholds, auc

def find_best_threshold_balanced(df):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []

    for threshold in thresholds:
        predictions = (df['cosine_similarity'] > threshold).astype(int)
        f1 = f1_score(df['unionable'], predictions)
        f1_scores.append(f1)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    return best_threshold, best_f1

def plot_youden_and_f1(df):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    youdens_index = []
    for threshold in thresholds:
        predictions = (df['cosine_similarity'] > threshold).astype(int)
        f1 = f1_score(df['unionable'], predictions)
        f1_scores.append(f1)
        tn, fp, fn, tp = confusion_matrix(df['unionable'], predictions).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        youden_index = sensitivity + specificity - 1
        youdens_index.append(youden_index)
    best_threshold_youden, best_youden_index = find_best_threshold_youden(df)
    best_threshold, best_misclassification_rate = find_best_threshold_balanced(df)
    print(f'Best Threshold based on Youden’s Index: {best_threshold_youden}')
    print(f'Best Youden’s Index: {best_youden_index}')
    print(f'Best Threshold based on f1: {best_threshold}')
    

if __name__ == "__main__":

    """
    ### CALCULATE THRESHOLD (simple) - TEST SET TOPIC EMBEDDING
    comparison_gt_file_path = "../results/test/topic-embedding/simple/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD (dtypes) - TEST SET TOPIC EMBEDDING
    comparison_gt_file_path = "../results/test/topic-embedding/dtypes/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD (dbpedia) - TEST SET TOPIC EMBEDDING
    comparison_gt_file_path = "../results/test/topic-embedding/dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD (dtypes-dbpedia) - TEST SET TOPIC EMBEDDING
    comparison_gt_file_path = "../results/test/topic-embedding/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD (simple) - TEST SET TOPIC FILTERING
    comparison_gt_file_path = "../results/test/topic-filtering/simple/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD (dtypes) - TEST SET TOPIC FILTERING
    comparison_gt_file_path = "../results/test/topic-filtering/dtypes/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD (dbpedia) - TEST SET TOPIC FILTERING
    comparison_gt_file_path = "../results/test/topic-filtering/dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""

    #"""
    ### CALCULATE THRESHOLD (dtypes-dbpedia) - TEST SET TOPIC FILTERING
    comparison_gt_file_path = "../results/test/topic-filtering/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""


    df = load_data(comparison_gt_file_path)
    fpr, tpr, thresholds, auc = plot_roc_curve(df)
    plot_youden_and_f1(df)

