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
from utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from compressors.arithmetic_coding import AECParams, ArithmeticEncoder, ArithmeticDecoder, FixedFreqModel
from core.data_block import DataBlock

def test_chow_liu_tree():
    # Generate a simple CSV file that has some degree of correlation b/w 2 columns
    # Age and height are somewhat correlated
    df = pd.DataFrame(columns=['Id','Age','Salary','Height'])
    df['Id'] = range(1, 26)
    age_data = np.random.randint(4, 40, 25)
    df['Age'] = age_data
    salary_data = np.random.randint(40000, 220000, 25)
    df['Salary'] = salary_data
    height_noise = 5*np.random.uniform(0, 1, 25)
    df['Height'] = 175 + height_noise
    df.loc[df['Age'] < 20, 'Height'] = 140 + (df['Age']-10)*6
    print(df)
    filepath = "../dataset/test_chow_liu_some_correlated.csv"
    df.to_csv(filepath, index=False)
      
    # Generate a CSV file with highly correlated columns
    # Salary is a function of age
    df = pd.DataFrame(columns=['Id','Age','Salary','Height'])
    df['Id'] = range(1, 26)
    age_data = np.random.randint(4, 40, 25)
    df['Age'] = age_data
    df['Salary'] = df['Age']*5000
    height_noise = 5*np.random.uniform(0, 1, 25)
    df['Height'] = 175 + height_noise
    df.loc[df['Age'] < 20, 'Height'] = 140 + (df['Age']-10)*6
    print(df)
    filepath = "../dataset/test_chow_liu_high_correlated.csv"
    df.to_csv(filepath, index=False)

# Parse over each column in the CSV file and create the ordering
# and dictionary 
def create_dict_ordering(data):
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

        # Build the ordering for this column
        # Use the dictionary that you created  
        for key, val in col_dict.items():
            data_ordering.loc[data[column]==val, column] = key  
    
    return column_dictionary, data_ordering
    

# Encode the dictionaries as plain text piped to gzip 
def encode_dictionary(dictionary):
    with open(out_filename, "ab") as f:
        dict_serialized = pickle.dumps(dictionary)
        dict_compressed = gzip.compress(bytes(dict_serialized))
        len_compressed = len(dict_compressed).to_bytes(8, sys.byteorder)
        f.write(len_compressed)
        f.write(dict_compressed)
        f.close()

# Create the marginal histogram for all columns
# We use a list to store the histograms for all columns
# Return the self-entropy list used to calculate mutual information later
# Encoding per column in order:
# len(support set) | len(encoded_hist_of_hist) | encoded_hist_of_hist | len(marginal_hist) | encoded_marginal_hist
def create_marginal_hist(ordered_data):
    marginal_hist_list = []
    self_entropy_list = []
    with open(out_filename, "ab") as f:
        for column in ordered_data:
            marginal_hist = ordered_data[column].value_counts(sort=False).to_dict()
            marginal_hist_list.append(marginal_hist)
            # Store the self entropy of this column in a list
            marginal_prob_dist = Frequencies(marginal_hist).get_prob_dist()
            self_entropy_list.append(marginal_prob_dist.entropy)

            support_set = marginal_hist.keys()
            frequencies = marginal_hist.values()

            # Find the histogram of histogram
            df_freq = pd.DataFrame(frequencies)
            freq_of_freq = df_freq.iloc[:, 0].value_counts().to_dict()

            # Encode the size of the support set in plain text
            f.write(len(support_set).to_bytes(8, sys.byteorder))
            
            # Encode the histogram of histogram (also called fingerprint) in plain text
            fingerprint_serialized = pickle.dumps(freq_of_freq)
            fingerprint_compressed = gzip.compress(bytes(fingerprint_serialized))
            len_compressed = len(fingerprint_compressed).to_bytes(8, sys.byteorder)
            f.write(len_compressed)
            f.write(fingerprint_compressed)

            # Encode the second column (frequencies) using AEC
            data_freq_set = DataBlock(frequencies)
            freq_freq_set = Frequencies(freq_of_freq)
            aec_encoder = ArithmeticEncoder(AECParams(), freq_freq_set, FixedFreqModel)
            aec_encoding = aec_encoder.encode_block(data_freq_set).tobytes()
            encoding_size = len(aec_encoding).to_bytes(8, sys.byteorder)
            f.write(encoding_size)
            f.write(aec_encoding)      
        
        f.close()

    return encoding_size, self_entropy_list


def create_pairwise_joint_hist(ordered_data, n_feat, self_entropy_list):
    # Create a pandas dataframe that contains tuples of pairwise combinations of columns
    num_pairs = n_feat*(n_feat-1)/2
    for index1 in range(n_feat):
        for index2 in range((index1+1), n_feat):
            pairwise_val = pd.Series(list(zip(ordered_data.iloc[:, index1], ordered_data.iloc[:, index2])))
            if index1==0 and index2==1:
                pairwise_columns = pd.DataFrame(data=pairwise_val, columns=[(index1, index2)])
            else:
                append_col = pd.DataFrame(data=pairwise_val, columns=[(index1, index2)])        
                pairwise_columns = pd.concat((pairwise_columns, append_col), axis=1)

    assert(len(pairwise_columns.columns) == num_pairs)
    print(pairwise_columns)

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

    # Encode the joint support set of each pairwise combination of columns
    # Use Golomb encoding 

    # Encode the pairwise joint histogram - using Arithmetic encoding


    # print(joint_entropy_list)
    # print(pairwise_mutual_info)
    return pairwise_mutual_info


# Construct the Chow-Liu tree
def construct_chow_liu_tree(pairwise_mutual_info):
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

    # Encode the Chow-Liu tree in bytes (plain text) for the decoder


def chow_liu_encoder(data, num_features):
    # Create the dictionary and ordered data from the read CSV
    dictionary, ordering = create_dict_ordering(data)

    # Encode the dictionary in plain text and write this to the compressed file
    # Encode this using gzip - | num_total_bytes_written_by_gzip | compressed_content_by_gzip |
    encode_dictionary(dictionary)

    # Create and encode the marginal histogram
    enc_size_marg_hist, self_entropy_list = create_marginal_hist(ordering)

    # Create and encode the pairwise joint histogram
    mutual_info = create_pairwise_joint_hist(ordering, num_features, self_entropy_list)

    # Compute the weight of the Chow-Liu tree

    # Generate the Chow-Liu tree
    construct_chow_liu_tree(mutual_info)

    # Perform pairwise encoding of columns in the dataset



###################################################################################################################
############### DECODER FUNCTIONS ##############################################################################

# Retrieve the dictionary back
# Returns the number of bytes already read from the file and the 
# decoded dictionary
def decode_dictionary():
    with open(out_filename, "rb") as f:
        len_bytes_to_read = int.from_bytes(f.read(8), sys.byteorder)
        data = f.read(len_bytes_to_read)
        dict_bytes = gzip.decompress(data)
        dict_deserialized = pickle.loads(dict_bytes)
        file_pos = f.tell()
        f.close()
        return file_pos, dict_deserialized
        


# Decode the marginal histogram for all columns
# We use a list to store the histograms for all columns
# Encoded column is:
# len(support set) | len(encoded_hist_of_hist) | encoded_hist_of_hist | len(marginal_hist) | encoded_marginal_hist
def decode_marginal_hist(pos_file, n_feat):
    marginal_hist_list = []
    with open(out_filename, "rb") as f:
        f.seek(pos_file)
        for col_idx in range(0, n_feat):
            len_support_set = int.from_bytes(f.read(8), sys.byteorder)
            support_set = list(range(len_support_set))
            
            # Decode the fingerprint
            len_fingerprint = int.from_bytes(f.read(8), sys.byteorder)
            data = f.read(len_fingerprint)
            fingerprint_bytes = gzip.decompress(data)
            fingerprint_deserialized = pickle.loads(fingerprint_bytes)
            
            # Use the decoded fingerprint to decode the frequencies of support set
            len_data_freq_set = int.from_bytes(f.read(8), sys.byteorder)
            data_freq_set = f.read(len_data_freq_set)
            data_freq_bits = BitArray()
            data_freq_bits.frombytes(data_freq_set)
            freq_freq_set = Frequencies(fingerprint_deserialized)
            aec_decoder = ArithmeticDecoder(AECParams(), freq_freq_set, FixedFreqModel)
            aec_decoding, bits_consumed = aec_decoder.decode_block(data_freq_bits)
            
            # Re-create the marginal histogram
            marginal_hist = dict(zip(support_set, aec_decoding.data_list))
            marginal_hist_list.append(marginal_hist)
        
        # print(marginal_hist_list)
        pos_file = f.tell()
        f.close()
    
    return pos_file, marginal_hist_list

def chow_liu_decoder(num_features):
    # Position to track the seeker in the compressed file
    pos_outfile = 0

    # Decode the dictionary from the compressed file
    pos_outfile, dictionary_all_cols = decode_dictionary()

    # Decode the marginal histogram 
    pos_outfile = decode_marginal_hist(pos_outfile, num_features)

   


if __name__ == "__main__":
    # Read the CSV file for our tabular database
    filename = sys.argv[1]
    filename = os.path.abspath(filename)
    # Skip the header row while reading the dataframe
    data = pd.read_csv(filename, skiprows=1, header=None)

    out_filename = filename + ".compressed"
    if os.path.exists(out_filename):
        os.remove(out_filename)
    
    num_features = len(data.columns)
    chow_liu_encoder(data, num_features)
    
    chow_liu_decoder(num_features)
