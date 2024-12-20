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


def extract_labels(file_path):
    """
    """
    labels = []
    g = Graph()
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    g.parse(file_path, format="turtle")
    for s in g.subjects(RDF.type, DSV.Column):
        label = next((str(o) for p, o in g.predicate_objects(s) if p == RDFS.label), None)
        labels.append(str(label))
    return labels

def calculate_embeddings(model, folder_path, output_file):
    """
    """
    all_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        labels = extract_labels(file_path)
        if not labels:
            continue
        embeddings = model.encode(labels, convert_to_tensor=False)
        all_data.append({
            'File': file_name,
            'Embeddings': [emb.tolist() for emb in embeddings]})
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":

    """
    ### CALCULATE EMBEDDINGS (simple data) - EVALUATION SET CANDIDATE TOPIC FILTERING
    folder_path = "../../../data/eval/simple/meta_datalake"
    output_file = "../../../results/eval/topic-filtering/simple/embeddings/candidate.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (simple data) - EVALUATION SET QUERY TOPIC FILTERING
    folder_path = "../../../data/eval/simple/meta_query"
    output_file = "../../../results/eval/topic-filtering/simple/embeddings/query.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (simple data) - TEST SET CANDIDATE TOPIC FILTERING
    folder_path = "../../../data/test/simple/meta_datalake"
    output_file = "../../../results/test/topic-filtering/simple/embeddings/candidate.csv"
    #"""

    #"""
    ### CALCULATE EMBEDDINGS (simple data) - TEST SET QUERY TOPIC FILTERING
    folder_path = "../../../data/test/simple/meta_query"
    output_file = "../../../results/test/topic-filtering/simple/embeddings/query.csv"
    #"""

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    calculate_embeddings(model, folder_path, output_file)