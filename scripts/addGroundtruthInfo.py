import pandas as pd
import sys

def make_df(file):
    """
    """
    df = pd.read_csv(file)
    return df

def compare_dfs(ss_df, gt_df):
    """
    """
    ss_df['query_table'] = ss_df['query'] + '.csv'
    ss_df['data_lake_table'] = ss_df['candidate'] + '.csv'
    comparison_df = pd.merge(
        ss_df,
        gt_df[['query_table', 'data_lake_table', 'unionable']],
        how='left',
        on=['query_table', 'data_lake_table']
    )
    comparison_df.drop(columns=['query_table', 'data_lake_table'], inplace=True)
    return comparison_df

def write_to_csv(comparison_df, comparison_outfile_path):
    """
    """
    comparison_df.to_csv(comparison_outfile_path, index=False)
    return

def main(ss_file, gt_file, comparison_outfile_path):
    """
    """
    ss_df = make_df(ss_file)
    gt_df = make_df(gt_file)
    comparison_df = compare_dfs(ss_df, gt_df)
    write_to_csv(comparison_df, comparison_outfile_path)
    return 

if __name__ == "__main__":

    """
    ### addGroundtruthInfo (simple data) - EVALUATION SET
    ss_file = "../results/eval/simple/cosine_similarity.csv"
    comparison_outfile_path = "../results/eval/simple/cosine_similarity_with_groundtruth.csv"
    """

    """
    ### addGroundtruthInfo (simple data) - TEST SET
    ss_file = "../results/test/simple/cosine_similarity.csv"
    comparison_outfile_path = "../results/test/simple/cosine_similarity_with_groundtruth.csv"
    """

    """
    ### addGroundtruthInfo (dtypes data) - EVALUATION SET
    ss_file = "../results/eval/enriched_dtypes/cosine_similarity.csv"
    comparison_outfile_path = "../results/eval/enriched_dtypes/cosine_similarity_with_groundtruth.csv"
    """

    #"""
    ### addGroundtruthInfo (dtypes data) - TEST SET
    ss_file = "../results/test/enriched_dtypes/cosine_similarity.csv"
    comparison_outfile_path = "../results/test/enriched_dtypes/cosine_similarity_with_groundtruth.csv"
    #"""

    gt_file = "../data/groundtruth.csv"
    main(ss_file, gt_file, comparison_outfile_path)