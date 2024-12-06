import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix


def read_comparison_gt_file(comparison_gt_file_path):
    """
    """
    df_comparison_gt = pd.read_csv(comparison_gt_file_path)
    return df_comparison_gt

def check_float_type(df_comparison_gt):
    """
    """
    df_comparison_gt['mean_similarity'] = df_comparison_gt['mean_similarity'].astype(float)
    df_comparison_gt['unionable'] = df_comparison_gt['unionable'].astype(int)
    return df_comparison_gt

def calculate_ROC_curve(df_comparison_gt):
    """
    """
    fpr, tpr, thresholds = roc_curve(df_comparison_gt['unionable'], df_comparison_gt['mean_similarity'])
    return fpr, tpr, thresholds

def calculate_AUC(df_comparison_gt):
    """
    """
    roc_auc = roc_auc_score(df_comparison_gt['unionable'], df_comparison_gt['mean_similarity'])
    print(f"Area Under the Curve (AUC): {roc_auc:.3f}")
    return roc_auc

def plot_ROC_curve(fpr, tpr, roc_auc):
    """
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    return 

def calculate_youdens_j(alpha, beta, tpr, fpr, thresholds):
    """
    """
    youden_index = alpha*tpr - beta*fpr
    optimal_idx = youden_index.argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    return optimal_threshold

def make_classification_report(df_comparison_gt, optimal_threshold):
    """
    """
    df_comparison_gt['predicted'] = (df_comparison_gt['mean_similarity'] >= optimal_threshold).astype(int)
    print(classification_report(df_comparison_gt['unionable'], df_comparison_gt['predicted']))
    return

def make_confusion_matrix(df_comparison_gt):
    """
    """
    conf_matrix = confusion_matrix(df_comparison_gt['unionable'], df_comparison_gt['predicted'])
    print("Confusion Matrix:")
    print(conf_matrix)
    return

def main(comparison_gt_file_path, alpha=1, beta=1):
    """
    """
    df_comparison_gt = read_comparison_gt_file(comparison_gt_file_path)
    df_comparison_gt = check_float_type(df_comparison_gt)
    fpr, tpr, thresholds = calculate_ROC_curve(df_comparison_gt)
    roc_auc = calculate_AUC(df_comparison_gt)
    plot_ROC_curve(fpr, tpr, roc_auc)
    optimal_threshold = calculate_youdens_j(alpha, beta, tpr, fpr, thresholds)
    make_classification_report(df_comparison_gt, optimal_threshold)
    make_confusion_matrix(df_comparison_gt)
    return 


if __name__ == "__main__":

    """
    ### CALCULATE THRESHOLD (simple data) - TEST SET
    comparison_gt_file_path = "../results/test/simple/cosine_similarity_with_groundtruth.csv"
    """

    #"""
    ### CALCULATE THRESHOLD (enriched dtypes data) - TEST SET
    comparison_gt_file_path = "../results/test/enriched_dtypes/cosine_similarity_with_groundtruth.csv"
    #"""

    alpha = 1.5
    beta = 1
    main(comparison_gt_file_path, alpha, beta)