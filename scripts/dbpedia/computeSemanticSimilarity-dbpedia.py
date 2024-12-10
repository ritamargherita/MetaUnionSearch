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


def extract_column_labels_and_dbpedia(g, DSV):
    """
    """
    column_labels_dbpedia = []
    for s in g.subjects(RDF.type, DSV.Column):
        label = next((str(o) for p, o in g.predicate_objects(s) if p == RDFS.label), None)
        col_dbpedia = next((str(o) for p, o in g.predicate_objects(s) if p == DSV.columnProperty), None)
        if label and col_dbpedia:
            column_labels_dbpedia.append((label, col_dbpedia))
    return column_labels_dbpedia

def compute_dataset_embedding(column_labels_dbpedia, model):
    """
    """
    embeddings = model.encode(column_labels_dbpedia)
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
    column_labels_dbpedia = extract_column_labels_and_dbpedia(g, DSV)
    embeddings = compute_dataset_embedding(column_labels_dbpedia, model)
    write_embeddings_csv(candidate_output_file, filename, embeddings)
    return

def make_embeddings_query(query_file_path, filename, model, query_output_file):
    """
    """
    g = Graph()
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    g.parse(query_file_path, format="turtle") 
    column_labels_dbpedia = extract_column_labels_and_dbpedia(g, DSV)
    embeddings = compute_dataset_embedding(column_labels_dbpedia, model)
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
                if query_embedding.size == 0:
                    print(query_file_path)
                    raise ValueError("query_embedding is empty. Please check your input or model encoding.")
                if query_embedding.ndim == 1:
                    query_embedding = query_embedding.reshape(1, -1)
                query_embedding = normalize(query_embedding, axis=1)

                similarity_matrix = cosine_similarity(candidate_embedding, query_embedding)
                mean_similarity = np.mean(similarity_matrix)
                csv_writer.writerow([query_filename.split('.')[0], candidate_filename.split('.')[0], mean_similarity])
    return
    

if __name__ == "__main__":

    """
    ### COMPUTE SEMANTIC SIMILARITY (dbpedia) - EVALUATION SET
    candidate_folder = "../../data/eval/enriched_dbpedia/meta_datalake"
    query_folder = "../../data/eval/enriched_dbpedia/meta_query"
    candidate_output_file = "../../results/eval/enriched_dbpedia/embeddings/candidate.csv"
    query_output_file = "../../results/eval/enriched_dbpedia/embeddings/query.csv"
    cosine_similarity_output_file = "../../results/eval/enriched_dbpedia/cosine_similarity.csv"
    """

    #"""
    ### COMPUTE SEMANTIC SIMILARITY (dbpedia) - TEST SET
    candidate_folder = "../../data/test/enriched_dbpedia/meta_datalake"
    query_folder = "../../data/test/enriched_dbpedia/meta_query"
    candidate_output_file = "../../results/test/enriched_dbpedia/embeddings/candidate.csv"
    query_output_file = "../../results/test/enriched_dbpedia/embeddings/query.csv"
    cosine_similarity_output_file = "../../results/test/enriched_dbpedia/cosine_similarity.csv"
    #"""

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    main(candidate_folder, query_folder, model, candidate_output_file, query_output_file, cosine_similarity_output_file)