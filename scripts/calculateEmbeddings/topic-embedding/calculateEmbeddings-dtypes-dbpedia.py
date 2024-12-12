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


def merge_graphs(folder1, folder2):
    """
    """
    merged_graphs = {}
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    common_files = files1.intersection(files2)
    for file_name in common_files:
        g1 = Graph()
        g2 = Graph()
        g1.parse(os.path.join(folder1, file_name), format="turtle")
        g2.parse(os.path.join(folder2, file_name), format="turtle")
        g1 += g2
        merged_graphs[file_name] = g1
    return merged_graphs


def extract_labels_and_types_and_topics(graph):
    """
    """
    labels = []
    topic = None
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    for s in graph.subjects(RDF.type, DSV.Dataset):
        topic = next((str(o) for p, o in graph.predicate_objects(s) if p == DCTERMS.subject), None)
        break
    for s in graph.subjects(RDF.type, DSV.Column):
        label = next((str(o) for p, o in graph.predicate_objects(s) if p == RDFS.label), None)
        prop_value = next((str(o) for p, o in graph.predicate_objects(s) if p == DSV.columnProperty), None)
        type_value = next((str(o) for p, o in graph.predicate_objects(s) if p == DCTERMS.type), None)
        if label and prop_value and type_value:
            labels.append(f"{label} {prop_value} {type_value}")
    return labels, topic

def calculate_embeddings(model, merged_graphs, output_file):
    """
    """
    all_data = []
    for file_name, graph in merged_graphs.items():
        labels, topic = extract_labels_and_types_and_topics(graph)
        if not labels:
            continue
        labels_embeddings = model.encode(labels, convert_to_tensor=False)
        topic_embedding = model.encode([topic], convert_to_tensor=False)[0] if topic else np.zeros_like(labels_embeddings[0])
        combined_embeddings = []
        for label_emb in labels_embeddings:
            combined_emb = 0.75 * np.array(label_emb) + 0.25 * np.array(topic_embedding)
            combined_embeddings.append(combined_emb)
        all_data.append({
            'File': file_name,
            'Embeddings': [emb.tolist() for emb in combined_embeddings]})
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - EVALUATION SET CANDIDATE TOPIC EMBEDDING
    folder_path_dbpedia = "../../../data/eval/dbpedia/meta_datalake"
    folder_path_dtypes = "../../../data/eval/dtypes/meta_datalake"
    output_file = "../../../results/eval/topic-embedding/dtypes-dbpedia/embeddings/candidate.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - EVALUATION SET QUERY TOPIC EMBEDDING
    folder_path_dbpedia = "../../../data/eval/dbpedia/meta_query"
    folder_path_dtypes = "../../../data/eval/dtypes/meta_query"
    output_file = "../../../results/eval/topic-embedding/dtypes-dbpedia/embeddings/query.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - TEST SET CANDIDATE TOPIC EMBEDDING
    folder_path_dbpedia = "../../../data/test/dbpedia/meta_datalake"
    folder_path_dtypes = "../../../data/test/dtypes/meta_datalake"
    output_file = "../../../results/test/topic-embedding/dtypes-dbpedia/embeddings/candidate.csv"
    #"""

    #"""
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - TEST SET QUERY TOPIC EMBEDDING
    folder_path_dbpedia = "../../../data/test/dbpedia/meta_query"
    folder_path_dtypes = "../../../data/test/dtypes/meta_query"
    output_file = "../../../results/test/topic-embedding/dtypes-dbpedia/embeddings/query.csv"
    #"""

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    merged_graphs = merge_graphs(folder_path_dbpedia, folder_path_dtypes)
    calculate_embeddings(model, merged_graphs, output_file)