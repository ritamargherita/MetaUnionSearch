import pandas as pd

def merge_similarity_with_groundtruth(cosine_file, groundtruth_file, output_file):
    cosine_df = pd.read_csv(cosine_file)
    groundtruth_df = pd.read_csv(groundtruth_file)
    cosine_df['query_table'] = cosine_df['query_table'].str.split('.').str[0]
    cosine_df['data_lake_table'] = cosine_df['data_lake_table'].str.split('.').str[0]
    groundtruth_df['query_table'] = groundtruth_df['query_table'].str.split('.').str[0]
    groundtruth_df['data_lake_table'] = groundtruth_df['data_lake_table'].str.split('.').str[0]
    groundtruth_dict = {
        (row['query_table'], row['data_lake_table']): row['unionable']
        for _, row in groundtruth_df.iterrows()}
    cosine_df['unionable'] = cosine_df.apply(
        lambda row: groundtruth_dict.get((row['query_table'], row['data_lake_table']), 0),
        axis=1)
    cosine_df.to_csv(output_file, index=False)


"""
### (simple) TOPIC EMBEDDING - EVAL
cosine_similarity_file = '../results/eval/topic-embedding/simple/cosine_similarities.csv'              
output_file = '../results/eval/topic-embedding/simple/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (simple) TOPIC EMBEDDING - TEST
cosine_similarity_file = '../results/test/topic-embedding/simple/cosine_similarities.csv'              
output_file = '../results/test/topic-embedding/simple/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes) TOPIC EMBEDDING - EVAL
cosine_similarity_file = '../results/eval/topic-embedding/dtypes/cosine_similarities.csv'              
output_file = '../results/eval/topic-embedding/dtypes/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dtypes) TOPIC EMBEDDING - TEST
cosine_similarity_file = '../results/test/topic-embedding/dtypes/cosine_similarities.csv'              
output_file = '../results/test/topic-embedding/dtypes/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dbpedia) TOPIC EMBEDDING - EVAL
cosine_similarity_file = '../results/eval/topic-embedding/dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-embedding/dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dbpedia) TOPIC EMBEDDING - TEST
cosine_similarity_file = '../results/test/topic-embedding/dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-embedding/dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes-dbpedia) TOPIC EMBEDDING - EVAL
cosine_similarity_file = '../results/eval/topic-embedding/dtypes-dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-embedding/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dtypes-dbpedia) TOPIC EMBEDDING - TEST
cosine_similarity_file = '../results/test/topic-embedding/dtypes-dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-embedding/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (simple) TOPIC FILTERING - EVAL
cosine_similarity_file = '../results/eval/topic-filtering/simple/cosine_similarities.csv'              
output_file = '../results/eval/topic-filtering/simple/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (simple) TOPIC FILTERING - TEST
cosine_similarity_file = '../results/test/topic-filtering/simple/cosine_similarities.csv'              
output_file = '../results/test/topic-filtering/simple/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes) TOPIC FILTERING - EVAL
cosine_similarity_file = '../results/eval/topic-filtering/dtypes/cosine_similarities.csv'              
output_file = '../results/eval/topic-filtering/dtypes/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dtypes) TOPIC FILTERING - TEST
cosine_similarity_file = '../results/test/topic-filtering/dtypes/cosine_similarities.csv'              
output_file = '../results/test/topic-filtering/dtypes/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dbpedia) TOPIC FILTERING - EVAL
cosine_similarity_file = '../results/eval/topic-filtering/dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-filtering/dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dbpedia) TOPIC FILTERING - TEST
cosine_similarity_file = '../results/test/topic-filtering/dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-filtering/dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes-dbpedia) TOPIC FILTERING - EVAL
cosine_similarity_file = '../results/eval/topic-filtering/dtypes-dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-filtering/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

#"""
### (dtypes-dbpedia) TOPIC FILTERING - TEST
cosine_similarity_file = '../results/test/topic-filtering/dtypes-dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-filtering/dtypes-dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

groundtruth_file = "../data/groundtruth.csv" 
merge_similarity_with_groundtruth(cosine_similarity_file, groundtruth_file, output_file)