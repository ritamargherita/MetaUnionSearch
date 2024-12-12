import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

def read_comparison_gt_file(comparison_gt_file_path):
    """
    """
    df_comparison_gt = pd.read_csv(comparison_gt_file_path)
    return df_comparison_gt

def check_float_type(df_comparison_gt):
    """
    """
    df_comparison_gt['cosine_similarity'] = pd.to_numeric(
        df_comparison_gt['cosine_similarity'], errors='coerce').fillna(0)
    df_comparison_gt['unionable'] = pd.to_numeric(
        df_comparison_gt['unionable'], errors='coerce').fillna(0)
    return df_comparison_gt

def calculate_ROC_curve(df_comparison_gt):
    """
    """
    fpr, tpr, thresholds = roc_curve(df_comparison_gt['unionable'], df_comparison_gt['cosine_similarity'])
    return fpr, tpr, thresholds

def calculate_AUC(df_comparison_gt):
    """
    """
    if len(df_comparison_gt['unionable'].unique()) < 2:
        print("Warning: Only one class present in 'unionable'. ROC AUC cannot be calculated.")
        return None
    roc_auc = roc_auc_score(df_comparison_gt['unionable'], df_comparison_gt['cosine_similarity'])
    print(f"Area Under the Curve (AUC): {roc_auc:.3f}")
    return roc_auc

def plot_ROC_curve(fpr, tpr, roc_auc):
    """
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line (chance level)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def calculate_youdens_j(alpha, beta, tpr, fpr, thresholds):
    """
    """
    youden_index = alpha * tpr - beta * fpr
    optimal_idx = youden_index.argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    return optimal_threshold

def make_classification_report(df_comparison_gt, optimal_threshold):
    """
    """
    df_comparison_gt['predicted'] = (df_comparison_gt['cosine_similarity'] >= optimal_threshold).astype(int)
    print(classification_report(df_comparison_gt['unionable'], df_comparison_gt['predicted']))

def make_confusion_matrix(df_comparison_gt):
    """
    """
    conf_matrix = confusion_matrix(df_comparison_gt['unionable'], df_comparison_gt['predicted'])
    print("Confusion Matrix:")
    print(conf_matrix)

def main(comparison_gt_file_path, alpha=1, beta=1):
    """
    """
    df_comparison_gt = read_comparison_gt_file(comparison_gt_file_path)
    df_comparison_gt = check_float_type(df_comparison_gt)
    if len(df_comparison_gt['unionable'].unique()) < 2:
        print("Error: Only one class present in the ground truth. Cannot compute ROC AUC or generate ROC curve.")
        return
    fpr, tpr, thresholds = calculate_ROC_curve(df_comparison_gt)
    roc_auc = calculate_AUC(df_comparison_gt)
    if roc_auc is None:
        return
    plot_ROC_curve(fpr, tpr, roc_auc)
    optimal_threshold = calculate_youdens_j(alpha, beta, tpr, fpr, thresholds)
    make_classification_report(df_comparison_gt, optimal_threshold)
    make_confusion_matrix(df_comparison_gt)

if __name__ == "__main__":

    """
    ### CALCULATE THRESHOLD TOPIC AGNOSTIC (simple data) - TEST SET
    comparison_gt_file_path = "../results/test/topic-agnostic/simple/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC AGNOSTIC (dtypes) - TEST SET
    comparison_gt_file_path = "../results/test/topic-agnostic/dtypes/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC AGNOSTIC (dbpedia) - TEST SET
    comparison_gt_file_path = "../results/test/topic-agnostic/dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC GUIDED (simple data) - TEST SET
    comparison_gt_file_path = "../results/test/topic-guided/simple/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC GUIDED (dtypes) - TEST SET
    comparison_gt_file_path = "../results/test/topic-guided/dtypes/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC GUIDED (dbpedia) - TEST SET
    comparison_gt_file_path = "../results/test/topic-guided/dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC DEPENDENT (simple) - TEST SET
    comparison_gt_file_path = "../results/test/topic-dependent/simple/cosine_similarities_with_groundtruth.csv"
    #"""

    """
    ### CALCULATE THRESHOLD TOPIC DEPENDENT (dtypes) - TEST SET
    comparison_gt_file_path = "../results/test/topic-dependent/dtypes/cosine_similarities_with_groundtruth.csv"
    #"""

    #"""
    ### CALCULATE THRESHOLD TOPIC DEPENDENT (dbpedia) - TEST SET
    comparison_gt_file_path = "../results/test/topic-dependent/dbpedia/cosine_similarities_with_groundtruth.csv"
    #"""

    alpha = 1
    beta = 1
    main(comparison_gt_file_path, alpha, beta)