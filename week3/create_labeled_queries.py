import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()
# stemmer = nltk.stem.snowball.SnowballStemmer("english")

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

def clean_query(query: str) -> str:
    # substitute_non_alnum
    clean_query = query.lower()
    clean_query = "".join([symbol if symbol.isalnum() else " " for symbol in clean_query])
    clean_query = re.sub(" +", " ", clean_query).strip()
    clean_query = " ".join([stemmer.stem(token) for token in clean_query.split()])

    return clean_query

assert clean_query("Beats By Dr. Dre- Monster Pro Over-the-Ear Headphones -") == "beat by dr dre monster pro over the ear headphon"
# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df["query"] = queries_df["query"].map(clean_query)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
category_mapper = dict(zip(parents_df["category"].values, parents_df["parent"].values))
while True:
    counts = queries_df["category"].value_counts()
    roll_up_categories = set(counts[counts < min_queries].index.values)
    if len(roll_up_categories) == 0:
        break
    else:
        # perform mapping to roll up
        _logger.info(f"{len(roll_up_categories)} categories below threshold = {min_queries}, rolling up ...")
        indexer = queries_df[queries_df["category"].isin(roll_up_categories)].index
        queries_df.loc[indexer, "category"] = queries_df.loc[indexer, "category"].map(category_mapper)

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
