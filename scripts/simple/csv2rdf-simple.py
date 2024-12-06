import pandas as pd
import os
import re
import sys
import logging
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import DCTERMS, RDFS


logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - Line %(lineno)d - %(message)s'
)

def make_input_file_df(input_file_path):
    """
    """
    input_file_df = pd.DataFrame()
    try:
        with open(input_file_path, 'r') as file:
            first_line = file.readline()
            is_header_blank = all(col.strip() == '' for col in first_line.split(';'))
        if is_header_blank:
            input_file_df = pd.read_csv(input_file_path, sep=';', header=None, index_col=None)
            input_file_df = input_file_df.iloc[:, 1:]
            input_file_df.columns = [f"Column_{i}" for i in range(1, len(input_file_df.columns) + 1)]
        else:
            input_file_df = pd.read_csv(input_file_path, sep=';', on_bad_lines='skip', index_col=None)
        input_file_df = input_file_df.loc[:, ~input_file_df.columns.str.contains('^Unnamed')]

    except pd.errors.ParserError as e:
        print(f"ParserError encountered: {e}")
        with open(input_file_path, 'r') as file:
            for line_no, line in enumerate(file, start=1):
                try:
                    pd.read_csv([line], sep=';')
                except Exception as line_error:
                    print(f"Error parsing line {line_no}: {line_error}")
    return input_file_df

def make_dataset_uri(filename):
    """
    """
    clean_filename = re.sub(r'.csv', '', filename)
    clean_filename = re.sub(r'[^a-zA-Z0-9]', '', clean_filename)
    clean_filename = re.sub(r'\\', '', clean_filename)
    clean_filename = re.sub(r'"', '', clean_filename)
    clean_filename = re.sub(r' ', '', clean_filename)
    dataset_uri = URIRef(f"http://metaUnionSearch/datasets/{clean_filename}")
    return dataset_uri

def make_dataset_schema_uri(dataset_uri):
    """
    """
    dataset_schema_uri = URIRef(dataset_uri+'/datasetSchema')
    return dataset_schema_uri

def make_column_uri(dataset_uri, column):
    """
    """
    clean_column = re.sub(r'[^a-zA-Z0-9]', '', column)
    clean_column = re.sub(r'\\', '', clean_column)
    clean_column = re.sub(r'"', '', clean_column)
    clean_column = re.sub(r' ', '_', clean_column)
    column_uri = URIRef(dataset_uri + f'/column/{clean_column}')
    return column_uri

def add_structural_layer_triples(g, DSV, MUS, dataset_uri, filename, dataset_schema_uri, input_file_df):
    """
    """
    dataset_uri = make_dataset_uri(filename)
    dataset_schema_uri = make_dataset_schema_uri(dataset_uri)

    g.add((dataset_uri, RDF.type, DSV.Dataset))
    g.add((dataset_uri, DCTERMS.title, Literal(filename)))
    g.add((dataset_uri, DCTERMS.subject, Literal(filename.split('_')[0].lower())))
    g.add((dataset_uri, DSV.datasetSchema, dataset_schema_uri))
    g.add((dataset_schema_uri, RDF.type, DSV.DatasetSchema)) 

    for column in input_file_df.columns:
        column_uri = make_column_uri(dataset_uri, column)
        g.add((dataset_schema_uri, DSV.column, column_uri))
        g.add((column_uri, RDF.type, DSV.Column))
        g.add((column_uri, RDFS.label, Literal(column)))   

    return g     

def add_triples(g, DSV, MUS, filename, input_file_df):
    """
    """
    dataset_uri = make_dataset_uri(filename)
    dataset_schema_uri = make_dataset_schema_uri(dataset_uri)
    add_structural_layer_triples(g, DSV, MUS, dataset_uri, filename, dataset_schema_uri, input_file_df)
    return g

def write_output_file(g, output_folder, filename):
    """
    """
    output_filename = f"{os.path.splitext(filename)[0]}.ttl"
    g.serialize(destination=os.path.join(output_folder, output_filename), format='turtle')
    return

def load_selected_topics(selected_topics_file):
    """
    """
    with open(selected_topics_file, 'r') as file:
        lines = file.readlines()
    selected_topics = [line.strip().replace(' ', '-') for line in lines[1:]]
    return selected_topics

def get_matching_files_for_topic(topic, folder):
    """
    """
    files_for_topics = [
        filename for filename in os.listdir(folder)
        if filename.startswith(topic) and filename.endswith(".csv")]
    return files_for_topics

def main(input_folder, output_folder, selected_topics_file):
    """
    """

    selected_topics = load_selected_topics(selected_topics_file)

    for topic in selected_topics:
        matching_files = get_matching_files_for_topic(topic, input_folder)

        for filename in matching_files:
            g = Graph()
            DSV = Namespace("https://w3id.org/dsv-ontology#")
            MUS = Namespace("http://metaUnionSearch/")   
            g.bind("dsv", DSV)
            g.bind("mus", MUS) 

            input_file_path = os.path.join(input_folder, filename)
            input_file_df = make_input_file_df(input_file_path)
            
            add_triples(g, DSV, MUS, filename, input_file_df)

            write_output_file(g, output_folder, filename)


if __name__ == "__main__":

    """
    ### CSV to RDF (simple data) - EVALUATION SET DATALAKE
    input_folder = "../alt-gen/data/ugen_v2/datalake"
    output_folder = "../data/eval/simple/meta_datalake"
    selected_topics_file = "../data/eval/topics_eval_set.txt"
    """

    """
    ### CSV to RDF (simple data) - EVALUATION SET QUERY
    input_folder = "../alt-gen/data/ugen_v2/query"
    output_folder = "../data/eval/simple/meta_query"
    selected_topics_file = "../data/eval/topics_eval_set.txt"
    """

    """
    ### CSV to RDF (simple data) - TEST SET DATALAKE
    input_folder = "../alt-gen/data/ugen_v2/datalake"
    output_folder = "../data/test/simple/meta_datalake"
    selected_topics_file = "../data/test/topics_test_set.txt"
    """

    ### CSV to RDF (simple data) - TEST SET QUERY
    input_folder = "../alt-gen/data/ugen_v2/query"
    output_folder = "../data/test/simple/meta_query"
    selected_topics_file = "/data/testtopics_test_set.txt"

    main(input_folder, output_folder, selected_topics_file)