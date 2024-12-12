import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def predict_unionability_on_threshold(input_file, output_file, threshold):
    """
    Predicts unionability based on the given threshold for cosine similarity.
    """
    df_input = pd.read_csv(input_file)
    df_input['cosine_similarity'] = pd.to_numeric(df_input['cosine_similarity'], errors='coerce')
    df_input['predicted_unionable'] = df_input['cosine_similarity'].apply(lambda x: 1 if x >= threshold else 0)
    df_input.to_csv(output_file, index=False)
    return

def compute_overall_accuracy(df_output):
    """
    Calculates overall accuracy for the predictions.
    """
    correct_predictions = (df_output['predicted_unionable'] == df_output['unionable']).sum()
    total_predictions = len(df_output)
    overall_accuracy = correct_predictions / total_predictions
    return overall_accuracy

def compute_zero_accuracy(df_output):
    """
    Calculates accuracy for class 0 (unionable = 0).
    """
    zero_predictions = df_output[df_output['unionable'] == 0]
    correct_zero_predictions = (zero_predictions['predicted_unionable'] == zero_predictions['unionable']).sum()
    zero_accuracy = correct_zero_predictions / len(zero_predictions) if len(zero_predictions) > 0 else 0
    return zero_accuracy

def compute_one_accuracy(df_output):
    """
    Calculates accuracy for class 1 (unionable = 1).
    """
    one_predictions = df_output[df_output['unionable'] == 1]
    correct_one_predictions = (one_predictions['predicted_unionable'] == one_predictions['unionable']).sum()
    one_accuracy = correct_one_predictions / len(one_predictions) if len(one_predictions) > 0 else 0
    return one_accuracy

def compute_tp_tn_fp_fn(df_output):
    """
    Calculates true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
    """
    tp = ((df_output['predicted_unionable'] == 1) & (df_output['unionable'] == 1)).sum()
    tn = ((df_output['predicted_unionable'] == 0) & (df_output['unionable'] == 0)).sum()
    fp = ((df_output['predicted_unionable'] == 1) & (df_output['unionable'] == 0)).sum()
    fn = ((df_output['predicted_unionable'] == 0) & (df_output['unionable'] == 1)).sum()
    return tp, tn, fp, fn

def compute_precision_recall_f1(a, b, c):
    """
    Computes precision, recall, and F1-score based on TP, FP, FN values.
    """
    precision = a / (a + b) if (a + b) > 0 else 0
    recall = a / (a + c) if (a + c) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_overall_precision(df_output):
    """
    Computes overall precision considering both classes.
    """
    tp, tn, fp, fn = compute_tp_tn_fp_fn(df_output)
    overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    return overall_precision

def print_confusion_matrix(df_output):
    """
    Prints the confusion matrix.
    """
    cm = confusion_matrix(df_output['unionable'], df_output['predicted_unionable'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(cmap='Blues')
    plt.show()
    return cm

def compute_stats(output_file):
    """
    Computes all statistics (accuracy, precision, recall, F1) and prints them.
    """
    df_output = pd.read_csv(output_file)

    overall_accuracy = compute_overall_accuracy(df_output)
    overall_precision = compute_overall_precision(df_output)
    tp, tn, fp, fn = compute_tp_tn_fp_fn(df_output)

    accuracy_0 = compute_zero_accuracy(df_output)
    precision_0, recall_0, f1_0 = compute_precision_recall_f1(tn, fn, fp)

    accuracy_1 = compute_one_accuracy(df_output)
    precision_1, recall_1, f1_1 = compute_precision_recall_f1(tp, fp, fn)

    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Overall Precision: {overall_precision:.2%}")
    print(f"Accuracy for '0' predictions: {accuracy_0:.2%}")
    print(f"Accuracy for '1' predictions: {accuracy_1:.2%}")
    print(f"Class '1' - Accuracy: {accuracy_1:.2%}, Precision: {precision_1:.2%}, Recall: {recall_1:.2%}, F1-Score: {f1_1:.2%}")
    print(f"Class '0' - Accuracy: {accuracy_0:.2%}, Precision: {precision_0:.2%}, Recall: {recall_0:.2%}, F1-Score: {f1_0:.2%}")

    # Print confusion matrix
    cm = print_confusion_matrix(df_output)
    print(f"Confusion Matrix:\n{cm}")
    
    return

def main(input_file, output_file, threshold):
    """
    Main function to predict unionability and compute statistics.
    """
    predict_unionability_on_threshold(input_file, output_file, threshold)
    compute_stats(output_file)
    return


if __name__ == "__main__":

    """
    ### META UNION SEARCH TOPIC EMBEDDING (simple)
    input_file = "../results/eval/topic-embedding/simple/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-embedding/simple/meta_union_search_with_groundtruth.csv"
    threshold = 0.8383838383838385
    #"""

    """
    ### META UNION SEARCH TOPIC EMBEDDING (dtypes)
    input_file = "../results/eval/topic-embedding/dtypes/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-embedding/dtypes/meta_union_search_with_groundtruth.csv"
    threshold = 0.8181818181818182
    #"""

    """
    ### META UNION SEARCH TOPIC EMBEDDING (dbpedia)
    input_file = "../results/eval/topic-embedding/dbpedia/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-embedding/dbpedia/meta_union_search_with_groundtruth.csv"
    threshold = 0.8080808080808082
    #"""

    """
    ### META UNION SEARCH TOPIC EMBEDDING (dtypes-dbpedia)
    input_file = "../results/eval/topic-embedding/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-embedding/dtypes-dbpedia/meta_union_search_with_groundtruth.csv"
    threshold = 0.787878787878788
    #"""

    """
    ### META UNION SEARCH TOPIC FILTERING (simple)
    input_file = "../results/eval/topic-filtering/simple/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-filtering/simple/meta_union_search_with_groundtruth.csv"
    threshold = 0.6666666666666667
    #"""

    """
    ### META UNION SEARCH TOPIC FILTERING (dtypes)
    input_file = "../results/eval/topic-filtering/dtypes/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-filtering/dtypes/meta_union_search_with_groundtruth.csv"
    threshold = 0.6262626262626263
    #"""

    """
    ### META UNION SEARCH TOPIC FILTERING (dbpedia)
    input_file = "../results/eval/topic-filtering/dbpedia/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-filtering/dbpedia/meta_union_search_with_groundtruth.csv"
    threshold = 0.686868686868687
    #"""

    #"""
    ### META UNION SEARCH TOPIC FILTERING (dtypes-dbpedia)
    input_file = "../results/eval/topic-filtering/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv"
    output_file = "../results/eval/topic-filtering/dtypes-dbpedia/meta_union_search_with_groundtruth.csv"
    threshold = 0.6565656565656566
    #"""


    main(input_file, output_file, threshold)