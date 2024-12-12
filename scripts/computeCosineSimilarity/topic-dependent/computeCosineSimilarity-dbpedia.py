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

def pad_embeddings(embedding, target_dim):
    """
    """
    padding_length = target_dim - embedding.shape[1]
    if padding_length > 0:
        padding = np.zeros((embedding.shape[0], padding_length))
        padded_embedding = np.concatenate([embedding, padding], axis=1)
    else:
        padded_embedding = embedding
    return padded_embedding

def calculate_cosine_similarities(query_df, candidate_df, output_file):
    """
    """
    results = []
    for _, query_row in query_df.iterrows():
        query_file = query_row['File']
        query_embedding = query_row['Embeddings']
        query_topic = query_file.split("_")[0]
        for _, candidate_row in candidate_df.iterrows():
            candidate_file = candidate_row['File']
            candidate_topic = candidate_file.split("_")[0]
            if query_topic == candidate_topic:
                candidate_embedding = candidate_row['Embeddings']
                target_dim = max(query_embedding.shape[0], candidate_embedding.shape[0])
                query_embedding_padded = pad_embeddings(query_embedding, target_dim)
                candidate_embedding_padded = pad_embeddings(candidate_embedding, target_dim)
                similarity = cosine_similarity(query_embedding_padded, candidate_embedding_padded)[0][0]
                results.append({
                    'query_table': query_file,
                    'data_lake_table': candidate_file,
                    'cosine_similarity': similarity
                })
    print(f"Saving results to {output_file}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print("Results saved successfully.")


if __name__ == "__main__":

    """
    ### COMPUTE SIMILARITY (dbpedia) - EVALUATION SET TOPIC DEPENDENT
    query_file_path = "../../../results/eval/topic-agnostic/dbpedia/embeddings/query.csv"     
    candidate_file_path = "../../../results/eval/topic-agnostic/dbpedia/embeddings/candidate.csv"
    output_file_path = "../../../results/eval/topic-dependent/dbpedia/cosine_similarities.csv"
    #"""

    #"""
    ### COMPUTE SIMILARITY (dbpedia) - TEST SET TOPIC DEPENDENT
    query_file_path = "../../../results/test/topic-agnostic/dbpedia/embeddings/query.csv"
    candidate_file_path = "../../../results/test/topic-agnostic/dbpedia/embeddings/candidate.csv"
    output_file_path = "../../../results/test/topic-dependent/dbpedia/cosine_similarities.csv"
    #"""


    query_embeddings = load_embeddings(query_file_path)
    candidate_embeddings = load_embeddings(candidate_file_path)
    calculate_cosine_similarities(query_embeddings, candidate_embeddings, output_file_path)