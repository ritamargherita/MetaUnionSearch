import os
import csv
import sys
import json
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import DCTERMS, RDFS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def extract_column_labels(g, DSV):
    """
    """
    column_labels = [str(o) for s, p, o in g
                     if p == RDFS.label and (s, RDF.type, DSV.Column) in g]
    return column_labels

def compute_dataset_embedding(column_labels, model):
    """
    """
    embeddings = model.encode(column_labels)
    return embeddings

def write_embeddings_csv(output_filepath, filename, embeddings):
    """
    """
    embeddings_str = json.dumps(embeddings.tolist())
    with open(output_filepath, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([filename, embeddings_str])
    return

def make_embeddings_candidate(candidate_file_path, filename, model, candidate_output_file):
    """
    """
    g = Graph()
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    g.parse(candidate_file_path, format="turtle") 
    column_labels = extract_column_labels(g, DSV)
    embeddings = compute_dataset_embedding(column_labels, model)
    write_embeddings_csv(candidate_output_file, filename, embeddings)
    return

def make_embeddings_query(query_file_path, filename, model, query_output_file):
    """
    """
    g = Graph()
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    g.parse(query_file_path, format="turtle") 
    column_labels = extract_column_labels(g, DSV)
    embeddings = compute_dataset_embedding(column_labels, model)
    write_embeddings_csv(query_output_file, filename, embeddings)
    return

def load_embeddings_from_csv(embeddings_csv_file):
    """
    """
    csv.field_size_limit(sys.maxsize)
    embeddings_dict = {}
    with open(embeddings_csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            filename = row[0]
            embeddings = json.loads(row[1])
            embeddings_dict[filename] = np.array(embeddings)
    return embeddings_dict

def average_embeddings(embedding_array):
    """
    """
    averaged_embeddings = np.mean(embedding_array)
    return averaged_embeddings

def compute_cosine_similarity(query_embedding, candidate_embeddings):
    """
    """
    similarities = {}
    for filename, candidate_embedding in candidate_embeddings.items():
        similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
        similarities[filename] = similarity
    return similarities

def main(candidate_folder, query_folder, model, candidate_output_file, query_output_file, cosine_similarity_output_file):
    """
    """
    for filename in os.listdir(candidate_folder):
        candidate_file_path = os.path.join(candidate_folder, filename)
        make_embeddings_candidate(candidate_file_path, filename, model, candidate_output_file)

    for filename in os.listdir(query_folder):
        query_file_path = os.path.join(query_folder, filename)
        make_embeddings_query(query_file_path, filename, model, query_output_file)

    candidate_embeddings = load_embeddings_from_csv(candidate_output_file)
    query_embeddings = load_embeddings_from_csv(query_output_file)

    with open(cosine_similarity_output_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["query", "candidate", "mean_similarity"])

        for query_filename, query_embedding in query_embeddings.items():
            prefix = query_filename.split('_')[0]
            matching_candidates = {fname: emb for fname, emb in candidate_embeddings.items() if fname.startswith(prefix)}
            
            for candidate_filename, candidate_embedding in matching_candidates.items():
                candidate_embedding = normalize(candidate_embedding, axis=1)
                query_embedding = normalize(query_embedding, axis=1)
                similarity_matrix = cosine_similarity(candidate_embedding, query_embedding)
                mean_similarity = np.mean(similarity_matrix)
                csv_writer.writerow([query_filename.split('.')[0], candidate_filename.split('.')[0], mean_similarity])

    return
    

if __name__ == "__main__":

    #"""
    ### COMPUTE SEMANTIC SIMILARITY (simple data) - EVALUATION SET 
    candidate_folder = "../../data/eval/simple/meta_datalake"
    query_folder = "../../data/eval/simple/meta_query"
    candidate_output_file = "../../results/eval/simple/embeddings/candidate.csv"
    query_output_file = "../../results/eval/simple/embeddings/query.csv"
    cosine_similarity_output_file = "../../results/eval/simple/cosine_similarity.csv"
    #"""

    """
    ### COMPUTE SEMANTIC SIMILARITY (simple data) - TEST SET
    candidate_folder = "../../data/test/simple/meta_datalake"
    query_folder = "../../data/test/simple/meta_query"
    candidate_output_file = "../../results/test/simple/embeddings/candidate.csv"
    query_output_file = "../../results/test/simple/embeddings/query.csv"
    cosine_similarity_output_file = "../../results/test/simple/cosine_similarity.csv"
    """

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    main(candidate_folder, query_folder, model, candidate_output_file, query_output_file, cosine_similarity_output_file)