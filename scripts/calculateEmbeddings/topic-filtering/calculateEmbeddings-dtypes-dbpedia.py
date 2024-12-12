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

def extract_labels_and_prop(g):
    """
    """
    combined_data = []
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    for s in g.subjects(RDF.type, DSV.Column):
        label = next((str(o) for p, o in g.predicate_objects(s) if p == RDFS.label), None)
        prop_value = next((str(o) for p, o in g.predicate_objects(s) if p == DSV.columnProperty), None)
        dtype = next((str(o) for p, o in g.predicate_objects(s) if p == DCTERMS.type), None)
        if label and prop_value and dtype:
            combined_data.append(f"{label} {prop_value} {dtype}")
    return combined_data

def calculate_embeddings(model, merged_graphs, output_file):
    """
    """
    all_data = []
    for file_name, g in merged_graphs.items():
        combined_data = extract_labels_and_prop(g)
        if not combined_data:
            continue
        embeddings = model.encode(combined_data, convert_to_tensor=False)
        all_data.append({
            'File': file_name,
            'Embeddings': [emb.tolist() for emb in embeddings]})
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - EVALUATION SET CANDIDATE TOPIC FILTERING
    folder_path_dbpedia = "../../../data/eval/dbpedia/meta_datalake"
    folder_path_dtypes = "../../../data/eval/dtypes/meta_datalake"
    output_file = "../../../results/eval/topic-filtering/dtypes-dbpedia/embeddings/candidate.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - EVALUATION SET QUERY TOPIC FILTERING
    folder_path_dbpedia = "../../../data/eval/dbpedia/meta_query"
    folder_path_dtypes = "../../../data/eval/dtypes/meta_query"
    output_file = "../../../results/eval/topic-filtering/dtypes-dbpedia/embeddings/query.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - TEST SET CANDIDATE TOPIC FILTERING
    folder_path_dbpedia = "../../../data/test/dbpedia/meta_datalake"
    folder_path_dtypes = "../../../data/test/dtypes/meta_datalake"
    output_file = "../../../results/test/topic-filtering/dtypes-dbpedia/embeddings/candidate.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (dtypes + dbpedia) - TEST SET QUERY TOPIC FILTERING
    folder_path_dbpedia = "../../../data/test/dbpedia/meta_query"
    folder_path_dtypes = "../../../data/test/dtypes/meta_query"
    output_file = "../../../results/test/topic-filtering/dtypes-dbpedia/embeddings/query.csv"
    #"""

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    merged_graphs = merge_graphs(folder_path_dbpedia, folder_path_dtypes)
    calculate_embeddings(model, merged_graphs, output_file)