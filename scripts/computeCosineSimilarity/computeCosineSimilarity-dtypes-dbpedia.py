import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(file_path):
    """
    """
    df = pd.read_csv(file_path)
    df['Embeddings'] = df['Embeddings'].apply(eval)
    df['Embeddings'] = df['Embeddings'].apply(np.array)
    return df

def compute_average_embedding(embeddings):
    """
    """
    return np.mean(embeddings, axis=0)

def calculate_cosine_similarities(query_df, candidate_df, output_file_path):
    """
    """
    results = []
    for _, query_row in query_df.iterrows():
        query_file = query_row['File']
        query_embedding = query_row['Embeddings']
        query_topic = query_file.split("_")[0]
        query_avg_embedding = compute_average_embedding(query_embedding)
        for _, candidate_row in candidate_df.iterrows():
            candidate_file = candidate_row['File']
            candidate_topic = candidate_file.split("_")[0]
            if query_topic == candidate_topic:
                candidate_embedding = candidate_row['Embeddings']
                candidate_avg_embedding = compute_average_embedding(candidate_embedding)
                similarity = cosine_similarity([query_avg_embedding], [candidate_avg_embedding])[0][0]
                results.append({
                    'query_table': query_file,
                    'data_lake_table': candidate_file,
                    'cosine_similarity': similarity
                })
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":

    """
    ### COMPUTE SIMILARITY (dtypes + dbpedia) - EVALUATION SET TOPIC EMBEDDING
    query_file_path = "../../results/eval/topic-embedding/dtypes-dbpedia/embeddings/query.csv"     
    candidate_file_path = "../../results/eval/topic-embedding/dtypes-dbpedia/embeddings/candidate.csv"
    output_file_path = "../../results/eval/topic-embedding/dtypes-dbpedia/cosine_similarities.csv"
    #"""

    """
    ### COMPUTE SIMILARITY (dtypes + dbpedia) - TEST SET TOPIC EMBEDDING
    query_file_path = "../../results/test/topic-embedding/dtypes-dbpedia/embeddings/query.csv"
    candidate_file_path = "../../results/test/topic-embedding/dtypes-dbpedia/embeddings/candidate.csv"
    output_file_path = "../../results/test/topic-embedding/dtypes-dbpedia/cosine_similarities.csv"
    #"""

    """
    ### COMPUTE SIMILARITY (dtypes + dbpedia) - EVALUATION SET TOPIC FILTERING
    query_file_path = "../../results/eval/topic-filtering/dtypes-dbpedia/embeddings/query.csv"     
    candidate_file_path = "../../results/eval/topic-filtering/dtypes-dbpedia/embeddings/candidate.csv"
    output_file_path = "../../results/eval/topic-filtering/dtypes-dbpedia/cosine_similarities.csv"
    #"""

    #"""
    ### COMPUTE SIMILARITY (dtypes + dbpedia) - TEST SET TOPIC FILTERING
    query_file_path = "../../results/test/topic-filtering/dtypes-dbpedia/embeddings/query.csv"
    candidate_file_path = "../../results/test/topic-filtering/dtypes-dbpedia/embeddings/candidate.csv"
    output_file_path = "../../results/test/topic-filtering/dtypes-dbpedia/cosine_similarities.csv"
    #"""


    query_embeddings = load_embeddings(query_file_path)
    candidate_embeddings = load_embeddings(candidate_file_path)
    calculate_cosine_similarities(query_embeddings, candidate_embeddings, output_file_path)