import os
import csv
import sys
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import DCTERMS, RDFS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    embeddings_sum = np.sum(embeddings, axis=0)
    return embeddings_sum

def compute_similarities(query_embedding, candidate_embeddings):
    """
    """
    cosine_similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
    return cosine_similarities

def load_topics(topics_file):
    """
    """
    with open(topics_file, 'r') as file:
        lines = file.readlines()
    topics = [line.strip().replace(' ', '-') for line in lines[1:]]
    return topics

def get_matching_files_for_topic(topic, folder):
    """
    """
    files_for_topic = [
        filename for filename in os.listdir(folder)
        if filename.startswith(topic) and filename.endswith(".ttl")]
    return files_for_topic

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

def process_candidate_file(g, DSV, candidate_filepath, topic_queries, model, output_filepath):
    """
    """
    candidate_name = os.path.splitext(os.path.basename(candidate_filepath))[0]
    column_labels = extract_column_labels(g, DSV)
    if column_labels:
        candidate_embedding = compute_dataset_embedding(column_labels, model)
        query_embeddings = []
        query_names = []
        for query_filepath in topic_queries:
            query_name = os.path.splitext(os.path.basename(query_filepath))[0]
            query_column_labels = extract_column_labels(query_filepath)
            if query_column_labels:
                query_embedding = compute_dataset_embedding(query_column_labels, model)
                query_embeddings.append(query_embedding)
                query_names.append(query_name)
        query_embeddings = np.array(query_embeddings)
        cosine_similarities = compute_similarities(candidate_embedding, query_embeddings)
        for query_name, score in zip(query_names, cosine_similarities):
            write_to_csv(output_filepath, query_name, candidate_name, score)
    return

def main(topics_file, candidate_folder, query_folder, model, output_filepath):
    """
    """
    g = Graph()
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    topics = load_topics(topics_file)

    with open(output_filepath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['query', 'candidate', 'score'])

    for topic in topics:
        topic_candidates = get_matching_files_for_topic(topic, candidate_folder)
        topic_queries = [
            os.path.join(query_folder, query_file)
            for query_file in get_matching_files_for_topic(topic, query_folder)]

        for candidate_file in topic_candidates:
            candidate_filepath = os.path.join(candidate_folder, candidate_file)
            g.parse(candidate_filepath, format="turtle") 
            process_candidate_file(g, DSV, candidate_filepath, topic_queries, model, output_filepath) 

    sort_csv_by_candidate_column(output_filepath)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python computeSemanticSimilarity.py <path_topics_file> <path_candidate_folder> <path_query_folder> <path_output_file>")
        sys.exit(1)

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    topics_file = sys.argv[1]
    candidate_folder = sys.argv[2]
    query_folder = sys.argv[3]
    output_filepath = sys.argv[4]

    main(topics_file, candidate_folder, query_folder, model, output_filepath)