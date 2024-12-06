import pandas as pd
import sys

def predict_unionability_on_threshold(input_file, output_file, threshold):
    """
    """
    df_input = pd.read_csv(input_file)
    df_input['predicted_unionable'] = df_input['mean_similarity'].apply(lambda x: 1 if x>= threshold else 0) #remember to change the threshold
    df_input.to_csv(output_file, index=False)
    return

def compute_overall_accuracy(df_output):
    """
    """
    correct_predictions = (df_output['predicted_unionable'] == df_output['unionable']).sum()
    total_predictions = len(df_output)
    overall_accuracy = correct_predictions / total_predictions
    return overall_accuracy

def compute_zero_accuracy(df_output):
    """
    """
    zero_predictions = df_output[df_output['unionable'] == 0]
    correct_zero_predictions = (zero_predictions['predicted_unionable'] == zero_predictions['unionable']).sum()
    zero_accuracy = correct_zero_predictions / len(zero_predictions) if len(zero_predictions) > 0 else 0
    return zero_accuracy

def compute_one_accuracy(df_output):
    """
    """
    one_predictions = df_output[df_output['unionable'] == 1]
    correct_one_predictions = (one_predictions['predicted_unionable'] == one_predictions['unionable']).sum()
    one_accuracy = correct_one_predictions / len(one_predictions) if len(one_predictions) > 0 else 0
    return one_accuracy

def compute_tp_tn_fp_fn(df_output):
    """
    """
    tp = ((df_output['predicted_unionable'] == 1) & (df_output['unionable'] == 1)).sum()
    tn = ((df_output['predicted_unionable'] == 0) & (df_output['unionable'] == 0)).sum()
    fp = ((df_output['predicted_unionable'] == 1) & (df_output['unionable'] == 0)).sum()
    fn = ((df_output['predicted_unionable'] == 0) & (df_output['unionable'] == 1)).sum()  
    return tp, tn, fp, fn 

def compute_precision_recall_f1(a, b, c):
    """
    """
    precision = a / (a + b) if (a + b) > 0 else 0
    recall = a / (a + c) if (a + c) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def compute_stats(output_file):
    """
    """
    df_output = pd.read_csv(output_file)

    overall_accuracy = compute_overall_accuracy(df_output)
    tp, tn, fp, fn = compute_tp_tn_fp_fn(df_output)

    accuracy_0 = compute_zero_accuracy(df_output)
    precision_0, recall_0, f1_0 = compute_precision_recall_f1(tn, fn, fp)

    accuracy_1 = compute_one_accuracy(df_output)
    precision_1, recall_1, f1_1 = compute_precision_recall_f1(tp, fp, fn)

    print(f"Accuracy: {overall_accuracy:.2%}") 
    print(f"Accuracy for '0' predictions: {accuracy_0:.2%}")
    print(f"Accuracy for '1' predictions: {accuracy_1:.2%}")
    print(f"Class '1' - Accuracy: {accuracy_1:.2%},Precision: {precision_1:.2%}, Recall: {recall_1:.2%}, F1-Score: {f1_1:.2%}")
    print(f"Class '0' - Accuracy: {accuracy_0:.2%}, Precision: {precision_0:.2%}, Recall: {recall_0:.2%}, F1-Score: {f1_0:.2%}")

    return

def main(input_file, output_file, threshold):
    """
    """
    predict_unionability_on_threshold(input_file, output_file, threshold)
    compute_stats(output_file)
    return


if __name__ == "__main__":

    """
    ### META UNION SEARCH (simple data) - EVAL SET
    input_file = "../results/eval/simple/cosine_similarity_with_groundtruth.csv"
    output_file = "../results/eval/simple/meta_union_search_with_groundtruth.csv"
    threshold = 0.177
    """

    #"""
    ### META UNION SEARCH (simple data) - EVAL SET
    input_file = "../results/eval/enriched_dtypes/cosine_similarity_with_groundtruth.csv"
    output_file = "../results/eval/enriched_dtypes/meta_union_search_with_groundtruth.csv"
    threshold = 0.175
    #"""

    main(input_file, output_file, threshold)