import os
import pandas as pd
import sys
import gzip
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any
from compressors.probability_models import (
    AdaptiveIIDFreqModel,
    AdaptiveOrderKFreqModel,
    FixedFreqModel,
    FreqModelBase,
)
from core.data_encoder_decoder import DataDecoder, DataEncoder
from core.prob_dist import Frequencies, ProbabilityDist

# Read the CSV file for our tabular database
filename = sys.argv[1]
filename = os.path.abspath(filename)
# Skip the header row while reading the dataframe
data = pd.read_csv(filename, skiprows=1, header=None)

# Parse over each column in the CSV file and create the ordering
# and dictionary 
# List of dictionaries for columns
column_dictionary = []
# New dataframe that represents the ordering for all the columns
data_ordering = data.copy(deep=True)
num_features = len(data_ordering.columns)

for column in data:
    # Build the dictionary for this column
    col_dict = {}
    column_list = data[column].unique().tolist()
    for i in range(len(column_list)):
        col_dict[i] = column_list[i]
    column_dictionary.append(col_dict)

    # print(col_dict)
    # Build the ordering for this column
    # Use the dictionary that you created  
    # print(data_ordering[column])
    for key, val in col_dict.items():
        data_ordering.loc[data[column]==val, column] = key  
    

# Encode the dictionaries as plain text piped to gzip 
# We will store this encoding in a separate file (for now)
# and share this file along with the primary file at the decoder stage
compressed_dict_path = os.path.join(os.getcwd(), "compressed_dict.txt.gz")
dict_serialized = pickle.dumps(column_dictionary)
with gzip.open(compressed_dict_path, 'wb') as f:
    if os.path.exists(compressed_dict_path):
        os.remove(compressed_dict_path)
    f.write(bytes(dict_serialized))

# Test to check if we are able to retrieve the dictionary back
# with gzip.open(compressed_dict_path, 'rb') as f:
#     dict_bytes = f.read()
# dict_deserialized = pickle.loads(dict_bytes)
# print(dict_deserialized)

# Create the marginal histogram for all columns
# We use a list to store the histograms for all columns
marginal_hist_list = []
self_entropy_list = []
for column in data_ordering:
    marginal_hist = data_ordering[column].value_counts().to_dict()
    marginal_hist_list.append(marginal_hist)
    # Store the self entropy of this column in a list
    marginal_prob_dist = Frequencies(marginal_hist).get_prob_dist()
    self_entropy_list.append(marginal_prob_dist.entropy)

    # Serialize this data to bytes
    # marginal_hist = pickle.dumps(marginal_hist)
    # Use Arithmetic Encoding to store this sequence



# Create the pairwise joint histogram of all columns (called pairwise_columns)
# This is a pandas dataframe that contains tuples of pairwise combinations of columns
num_pairs = num_features*(num_features-1)/2
for index1 in range(num_features):
    for index2 in range((index1+1), num_features):
        pairwise_val = pd.Series(list(zip(data_ordering.iloc[:, index1], data_ordering.iloc[:, index2])))
        if index1==0 and index2==1:
            pairwise_columns = pd.DataFrame(data=pairwise_val, columns=[(index1, index2)])
        else:
            append_col = pd.DataFrame(data=pairwise_val, columns=[(index1, index2)])        
            pairwise_columns = pd.concat((pairwise_columns, append_col), axis=1)

assert(len(pairwise_columns.columns) == num_pairs)

# Now obtain the pairwise joint histogram for each pairwise combination
# Also Calculate mutual information for all pairs of columns
# I(X; Y) = H(X) + H(Y) − H(X, Y)
# For now, this is the weight that we use for the edges of the Chow-Liu tree
pairwise_hist_list = []
joint_entropy_list = {}
pairwise_mutual_info = {}
for column in pairwise_columns:
    (index1, index2) = column
    pairwise_hist = pairwise_columns[column].value_counts().to_dict()
    pairwise_hist_list.append(pairwise_hist)
    pairwise_prob_dist = Frequencies(pairwise_hist).get_prob_dist()
    joint_entropy_list[(index1, index2)] = pairwise_prob_dist.entropy

    mutual_info = self_entropy_list[index1] + self_entropy_list[index2] - joint_entropy_list[(index1, index2)]
    pairwise_mutual_info[(index1, index2)] = mutual_info

assert(len(joint_entropy_list) == num_pairs)
assert(len(pairwise_mutual_info) == num_pairs)


# Construct the Chow-Liu tree
G = nx.Graph()
# Every vertex in this graph is a feature in the original tabular data
for x1 in range(num_features):
    G.add_node(x1)
    for x2 in range(x1):
        G.add_edge(x2, x1, weight=-(pairwise_mutual_info[(x2, x1)]))
chow_liu_tree = nx.minimum_spanning_tree(G)

# Generate the adjanceny list
print(chow_liu_tree.edges.data())
# Print the Chow-Liu tree
nx.draw(chow_liu_tree, with_labels=True)
plt.show()
