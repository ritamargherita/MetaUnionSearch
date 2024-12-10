import os
import re
import sys
import time
import json
import numpy as np
import csv

from dotenv import load_dotenv

from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch


def gpt_client(OPENAI_API_KEY):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def load_dbpedia_properties(dbpedia_file_path):
    with open(dbpedia_file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
def make_property_embeddings(dbpedia_properties, model):
        property_texts = [
            f"{prop['label']} - {prop['desc']}" for prop in dbpedia_properties
        ]
        property_embeddings = model.encode(property_texts, convert_to_tensor=True)    
        return property_embeddings

def make_header_embeddings(column_headers, model):
    header_embeddings = model.encode(column_headers, convert_to_tensor=True)
    return header_embeddings

def get_k_top_properties(column_headers, header_embeddings, property_embeddings, top_k, dbpedia_properties):
    top_k_properties = []
    for i, header in enumerate(column_headers):
        similarities = util.pytorch_cos_sim(header_embeddings[i], property_embeddings)[0]
        top_k_indices = torch.topk(similarities, k=top_k).indices
        for idx in top_k_indices:
            top_k_properties.append(dbpedia_properties[idx.item()])
    return top_k_properties

def get_response(client, updated_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": updated_prompt
            }
        ],
        response_format={ "type": "json_object" }
    )
    response = completion.choices[0].message.content
    response = response.replace("```json", "")
    response = response.replace("```", "")
    response = response.replace("\\", "")
    return response

def main(OPENAI_API_KEY, dbpedia_file_path, candidate_folder, query_folder, candidate_output_folder, query_output_folder, top_k, model, prompt):
    
    client = gpt_client(OPENAI_API_KEY)

    """
    for query_file in os.listdir(query_folder):
        file_path = os.path.join(query_folder, query_file)
        with open(file_path, 'r') as infile:
            reader = csv.reader(infile, delimiter=';')
            column_headers = next(reader)[1:]
            dbpedia_properties = load_dbpedia_properties(dbpedia_file_path)
            property_embeddings = make_property_embeddings(dbpedia_properties, model)
            header_embeddings = make_header_embeddings(column_headers, model)
            top_k_properties = get_k_top_properties(column_headers, header_embeddings, property_embeddings, top_k, dbpedia_properties)
            updated_prompt = prompt.format(column_headers_list=column_headers, dbpedia_properties=top_k_properties)
            response = get_response(client, updated_prompt)
            print(response)
            query_output_file = os.path.join(query_output_folder, query_file)
            query_output_file = query_output_file.replace('.csv', '.json')
            with open(query_output_file, "w") as file:
                file.write(response)

    """
    for candidate_file in os.listdir(candidate_folder):
        file_path = os.path.join(candidate_folder, candidate_file)
        with open(file_path, 'r') as infile:
            reader = csv.reader(infile, delimiter=';')
            column_headers = next(reader)[1:]
            dbpedia_properties = load_dbpedia_properties(dbpedia_file_path)
            property_embeddings = make_property_embeddings(dbpedia_properties, model)
            header_embeddings = make_header_embeddings(column_headers, model)
            top_k_properties = get_k_top_properties(column_headers, header_embeddings, property_embeddings, top_k, dbpedia_properties)
            updated_prompt = prompt.format(column_headers_list=column_headers, dbpedia_properties=top_k_properties)
            response = get_response(client, updated_prompt)
            print(response)
            candidate_output_file = os.path.join(candidate_output_folder, candidate_file)
            candidate_output_file = candidate_output_file.replace('.csv', '.json')
            with open(candidate_output_file, "w") as file:
                file.write(response)
    

    return


if __name__ == "__main__":

    dbpedia_file_path = "../data/dbpedia.json"
    candidate_folder = "../alt-gen/data/ugen_v2/datalake"
    query_folder = "../alt-gen/data/ugen_v2/query"
    candidate_output_folder = "../data/enrichments_dbpedia/datalake"
    query_output_folder = "../data/enrichments_dbpedia/query"

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    top_k = 5
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    prompt = """
    Map each of the following column headers to one of the DBpedia property provided below.

    Return ONLY the response, no other text.
    ### Response in JSON Format ###
    {{"mapping": ["column_header": "<the original column header>", "dbpedia_property_id": "<the DBpedia property ID>"]}}

    ### Column Headers ###
    {column_headers_list}

    ### DBpedia Properties ###
    {dbpedia_properties}
    """

    main(OPENAI_API_KEY, dbpedia_file_path, candidate_folder, query_folder, candidate_output_folder, query_output_folder, top_k, model, prompt)
