import os
import csv
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import DCTERMS, RDFS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_column_labels(ttl_file):
    """
    """
    g = Graph()
    g.parse(ttl_file, format="turtle")
    column_labels = [str(o) for s, p, o in g if p == RDFS.label]
    return column_labels

def compute_dataset_embedding(column_labels, model):
    """
    """
    embeddings = model.encode(column_labels)
    return np.sum(embeddings, axis=0)

def compute_similarities(query_embedding, candidate_embeddings):
    """
    """
    return cosine_similarity([query_embedding], candidate_embeddings)[0]

def get_matching_queries(candidate_name, query_folder):
    """
    """
    candidate_prefix = candidate_name.split('_')[0]
    matching_queries = [
        filename for filename in os.listdir(query_folder)
        if filename.startswith(candidate_prefix) and filename.endswith(".ttl")
    ]
    return matching_queries

def write_to_csv(output_filepath, query_name, candidate_name, score):
    """
    """
    with open(output_filepath, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([query_name, candidate_name, score])
    return

def sort_csv_by_candidate_column(output_filepath):
    """
    """
    with open(output_filepath, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        rows = list(csv_reader)

    header = rows[0]
    rows = rows[1:]
    rows.sort(key=lambda x: x[1])

    with open(output_filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(rows)
    
    return

def process_candidate_file(candidate_filepath, query_folder, model, output_filepath):
    """
    """
    candidate_name = os.path.splitext(os.path.basename(candidate_filepath))[0]
    column_labels = extract_column_labels(candidate_filepath)
    if column_labels:
        candidate_embedding = compute_dataset_embedding(column_labels, model)
        matching_queries = get_matching_queries(candidate_name, query_folder)
        if not matching_queries:
            return
        query_embeddings = []
        query_names = []
        for query_filename in matching_queries:
            query_filepath = os.path.join(query_folder, query_filename)
            query_name = os.path.splitext(query_filename)[0]
            query_column_labels = extract_column_labels(query_filepath)
            if query_column_labels:
                query_embedding = compute_dataset_embedding(query_column_labels, model)
                query_embeddings.append(query_embedding)
                query_names.append(query_name)
        query_embeddings = np.array(query_embeddings)
        similarities = compute_similarities(candidate_embedding, query_embeddings)
        for query_name, score in zip(query_names, similarities):
            write_to_csv(output_filepath, query_name, candidate_name, score)
    return

def main(candidate_folder, query_folder, model, output_filepath):
    """
    """
    with open(output_filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['query', 'candidate', 'score'])
    for candidate_filename in os.listdir(candidate_folder):
        if candidate_filename.endswith(".ttl"):
            candidate_filepath = os.path.join(candidate_folder, candidate_filename)
            process_candidate_file(candidate_filepath, query_folder, model, output_filepath)
    sort_csv_by_candidate_column(output_filepath)
    return


if __name__ == "__main__":
    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)
    candidate_folder = '../data/metadata/simple/metadata_lake'
    query_folder = '../data/metadata/simple/metaquery_lake'
    output_filepath = '../results/similarities_results.csv'
    main(candidate_folder, query_folder, model, output_filepath)