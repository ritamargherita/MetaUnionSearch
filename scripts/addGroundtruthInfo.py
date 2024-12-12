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
### (simple) TOPIC AGNOSTIC - EVAL
cosine_similarity_file = '../results/eval/topic-agnostic/simple/cosine_similarities.csv'              
output_file = '../results/eval/topic-agnostic/simple/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (simple) TOPIC AGNOSTIC - TEST
cosine_similarity_file = '../results/test/topic-agnostic/simple/cosine_similarities.csv'              
output_file = '../results/test/topic-agnostic/simple/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes) TOPIC AGNOSTIC - EVAL
cosine_similarity_file = '../results/eval/topic-agnostic/dtypes/cosine_similarities.csv'              
output_file = '../results/eval/topic-agnostic/dtypes/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dtypes) TOPIC AGNOSTIC - TEST
cosine_similarity_file = '../results/test/topic-agnostic/dtypes/cosine_similarities.csv'              
output_file = '../results/test/topic-agnostic/dtypes/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dbpedia) TOPIC AGNOSTIC - EVAL
cosine_similarity_file = '../results/eval/topic-agnostic/dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-agnostic/dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dbpedia) TOPIC AGNOSTIC - TEST
cosine_similarity_file = '../results/test/topic-agnostic/dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-agnostic/dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (simple) TOPIC GUIDED - EVAL
cosine_similarity_file = '../results/eval/topic-guided/simple/cosine_similarities.csv'              
output_file = '../results/eval/topic-guided/simple/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (simple) TOPIC GUIDED - TEST
cosine_similarity_file = '../results/test/topic-guided/simple/cosine_similarities.csv'              
output_file = '../results/test/topic-guided/simple/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes) TOPIC GUIDED - EVAL
cosine_similarity_file = '../results/eval/topic-guided/dtypes/cosine_similarities.csv'              
output_file = '../results/eval/topic-guided/dtypes/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dtypes) TOPIC GUIDED - TEST
cosine_similarity_file = '../results/test/topic-guided/dtypes/cosine_similarities.csv'              
output_file = '../results/test/topic-guided/dtypes/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dbpedia) TOPIC GUIDED - EVAL
cosine_similarity_file = '../results/eval/topic-guided/dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-guided/dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dbpedia) TOPIC GUIDED - TEST
cosine_similarity_file = '../results/test/topic-guided/dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-guided/dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (simple) TOPIC DEPENDENT - EVAL
cosine_similarity_file = '../results/eval/topic-dependent/simple/cosine_similarities.csv'              
output_file = '../results/eval/topic-dependent/simple/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (simple) TOPIC DEPENDENT - TEST
cosine_similarity_file = '../results/test/topic-dependent/simple/cosine_similarities.csv'              
output_file = '../results/test/topic-dependent/simple/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dtypes) TOPIC DEPENDENT - EVAL
cosine_similarity_file = '../results/eval/topic-dependent/dtypes/cosine_similarities.csv'              
output_file = '../results/eval/topic-dependent/dtypes/cosine_similarities_with_groundtruth.csv'    
#"""

"""
### (dtypes) TOPIC DEPENDENT - TEST
cosine_similarity_file = '../results/test/topic-dependent/dtypes/cosine_similarities.csv'              
output_file = '../results/test/topic-dependent/dtypes/cosine_similarities_with_groundtruth.csv'
#"""

"""
### (dbpedia) TOPIC DEPENDENT - EVAL
cosine_similarity_file = '../results/eval/topic-dependent/dbpedia/cosine_similarities.csv'              
output_file = '../results/eval/topic-dependent/dbpedia/cosine_similarities_with_groundtruth.csv'    
#"""

#"""
### (dbpedia) TOPIC DEPENDENT - TEST
cosine_similarity_file = '../results/test/topic-dependent/dbpedia/cosine_similarities.csv'              
output_file = '../results/test/topic-dependent/dbpedia/cosine_similarities_with_groundtruth.csv'
#"""

groundtruth_file = "../data/groundtruth.csv" 
merge_similarity_with_groundtruth(cosine_similarity_file, groundtruth_file, output_file)