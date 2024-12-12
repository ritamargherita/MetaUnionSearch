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

def extract_labels_and_topics(file_path):
    """
    """
    labels = []
    topic = None
    g = Graph()
    DSV = Namespace("https://w3id.org/dsv-ontology#")
    g.parse(file_path, format="turtle")
    for s in g.subjects(RDF.type, DSV.Dataset):
        topic = next((str(o) for p, o in g.predicate_objects(s) if p == DCTERMS.subject), None)
        break
    for s in g.subjects(RDF.type, DSV.Column):
        label = next((str(o) for p, o in g.predicate_objects(s) if p == RDFS.label), None)
        if label:
            labels.append(str(label))
    return labels, topic

def calculate_embeddings(model, folder_path, output_file):
    """
    """
    all_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        labels, topic = extract_labels_and_topics(file_path)
        label_embeddings = model.encode(labels, convert_to_tensor=False)
        topic_embedding = model.encode([topic], convert_to_tensor=False)[0]
        combined_embeddings = []
        for label_emb in label_embeddings:
            combined_emb = 0.75 * np.array(label_emb) + 0.25 * np.array(topic_embedding)
            combined_embeddings.append(combined_emb)
        all_data.append({
            'File': file_name,
            'Embeddings': [emb.tolist() for emb in combined_embeddings]})
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":

    """
    ### CALCULATE EMBEDDINGS (simple) - EVALUATION SET CANDIDATE TOPIC EMBEDDING
    folder_path = "../../../data/eval/simple/meta_datalake"
    output_file = "../../../results/eval/topic-embedding/simple/embeddings/candidate.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (simple) - EVALUATION SET QUERY TOPIC EMBEDDING
    folder_path = "../../../data/eval/simple/meta_query"
    output_file = "../../../results/eval/topic-embedding/simple/embeddings/query.csv"
    #"""

    """
    ### CALCULATE EMBEDDINGS (simple) - TEST SET CANDIDATE TOPIC EMBEDDING
    folder_path = "../../../data/test/simple/meta_datalake"
    output_file = "../../../results/test/topic-embedding/simple/embeddings/candidate.csv"
    #"""

    #"""
    ### CALCULATE EMBEDDINGS (simple) - TEST SET QUERY TOPIC EMBEDDING
    folder_path = "../../../data/test/simple/meta_query"
    output_file = "../../../results/test/topic-embedding/simple/embeddings/query.csv"
    #"""

    embedding_model = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(embedding_model)

    calculate_embeddings(model, folder_path, output_file)