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
    df_comparison_gt['score'] = df_comparison_gt['score'].astype(float)
    df_comparison_gt['unionable'] = df_comparison_gt['unionable'].astype(int)
    return df_comparison_gt

def calculate_ROC_curve(df_comparison_gt):
    """
    """
    fpr, tpr, thresholds = roc_curve(df_comparison_gt['unionable'], df_comparison_gt['score'])
    return fpr, tpr, thresholds

def calculate_AUC(df_comparison_gt):
    """
    """
    roc_auc = roc_auc_score(df_comparison_gt['unionable'], df_comparison_gt['score'])
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
    df_comparison_gt['predicted'] = (df_comparison_gt['score'] >= optimal_threshold).astype(int)
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
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python thresholdCalculator.py <comparison_gt_file_path> [<alpha_tpr> <beta_fpr>]")
        sys.exit(1)
    comparison_gt_file_path = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1
    beta = float(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(comparison_gt_file_path, alpha, beta)