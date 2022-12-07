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
from compressors.golomb_coder import GolombCodeParams, GolombUintEncoder, GolombUintDecoder
from core.data_block import DataBlock
from collections import OrderedDict

def test_chow_liu_tree():
    # Generate a simple CSV file that has some degree of correlation b/w 2 columns
    # Age and height are somewhat correlated
    df = pd.DataFrame(columns=['x1','x2','x3','x4'])
    df['x1'] = range(1, 26)
    df['x2'] = df['x1']
    df['x3'] = np.random.randint(40000, 220000, 25)
    df['x4'] = df['x3']
    filepath = "../dataset/test_chow_liu_some_correlated.csv"
    df.to_csv(filepath, index=False)
      
    # Generate a CSV file with highly correlated columns
    # Salary is a function of age
    # df = pd.DataFrame(columns=['Id','Age','Salary','Height'])
    # df['Id'] = range(1, 26)
    # age_data = np.random.randint(4, 40, 25)
    # df['Age'] = age_data
    # df['Salary'] = df['Age']*5000
    # height_noise = 5*np.random.uniform(0, 1, 25)
    # df['Height'] = 175 + height_noise
    # df.loc[df['Age'] < 20, 'Height'] = 140 + (df['Age']-10)*6
    # print(df)
    # filepath = "../dataset/test_chow_liu_high_correlated.csv"
    # df.to_csv(filepath, index=False)

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
# len(support set) | len(encoded_hist_of_hist) | encoded_hist_of_hist | len(encoded_marginal_hist) | encoded_marginal_hist
def create_marginal_hist(ordered_data):
    marginal_hist_list = []
    self_entropy_list = []
    storage_cost = 0
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
            freq_of_freq = df_freq.iloc[:, 0].value_counts(sort=False).to_dict()

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
            params = AECParams()
            freq_model_enc = AdaptiveIIDFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
            aec_encoder = ArithmeticEncoder(AECParams(), freq_model_enc)
            aec_encoding = aec_encoder.encode_block(data_freq_set).tobytes()
            encoding_size = len(aec_encoding)
            storage_cost += len(aec_encoding) 
            encoding_size = encoding_size.to_bytes(8, sys.byteorder)
            f.write(encoding_size)
            f.write(aec_encoding)
                 
        
        f.close()
    # print(marginal_hist_list)
    return encoding_size, self_entropy_list, storage_cost

# Create the pairwise joint histogram for all columns
# Encode both the support set and the frequencies using AEC
# Tried encoding the support set with Golomb coding - faced accuracy issues
def create_pairwise_joint_hist(ordered_data, n_feat, self_entropy_list, col_dict):
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
        pairwise_hist = pairwise_columns[column].value_counts(sort=False).to_dict()
        pairwise_hist = OrderedDict(sorted(pairwise_hist.items()))
        print(pairwise_hist)
        pairwise_hist_list.append(pairwise_hist)
        pairwise_prob_dist = Frequencies(pairwise_hist).get_prob_dist()
        joint_entropy_list[(index1, index2)] = pairwise_prob_dist.entropy

        mutual_info = self_entropy_list[index1] + self_entropy_list[index2] - joint_entropy_list[(index1, index2)]
        pairwise_mutual_info[(index1, index2)] = mutual_info

    assert(len(joint_entropy_list) == num_pairs)
    assert(len(pairwise_mutual_info) == num_pairs)

    # Encode the joint support set and frequencies
    with open(out_filename, "ab") as f:
        storage_cost = 0
        for column in pairwise_columns:
            (index1, index2) = column
            col_idx = pairwise_columns.columns.get_loc(column)
            # Get the dimensions of the joint binary support set 
            m = len(col_dict[index1])
            n = len(col_dict[index2])
            joint_support_set = np.zeros((m, n))

            # 1s in the indices (i, j) will correspond to that tuple being present in the support set
            column_list = pairwise_hist_list[col_idx].keys()
            print(column_list)
            for (idx1, idx2) in column_list:
                joint_support_set[idx1][idx2]=1
            
            # print(column_list)
            # print(joint_support_set)

            # Compute the distance between adjacent 1s and store in a list
            # This comprises the input data to the AEC encoder
            # Flatten the 2D numpy array into 1D for ease
            joint_support_set_1d = joint_support_set.flatten()
            f.write(len(joint_support_set_1d).to_bytes(8, sys.byteorder))
            idx_true = np.where(joint_support_set_1d)[0]
            idx_true = idx_true.tolist()
            # Distances will always be of the following form
            # Idx of the first 1 followed by distances between adjacent 1s
            dist_arr = [idx_true[0]]
            dist_arr.extend(np.diff(idx_true).tolist())
            dist_arr_data = DataBlock(dist_arr)
            dist_arr_freq = dist_arr_data.get_counts(order=0)

            # Encode the frequency dist. of support set in plain text
            dist_arr_freq_serialized = pickle.dumps(dist_arr_freq)
            dist_arr_freq_compressed = gzip.compress(bytes(dist_arr_freq_serialized))
            len_compressed = len(dist_arr_freq_compressed)
            storage_cost += len_compressed*8
            len_compressed = len_compressed.to_bytes(8, sys.byteorder)
            f.write(len_compressed)
            f.write(dist_arr_freq_compressed)
            

            # Encode the joint support set using AEC
            dist_freq = Frequencies(dist_arr_freq)
            params = AECParams()
            freq_model_enc = AdaptiveIIDFreqModel(dist_freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
            aec_encoder = ArithmeticEncoder(AECParams(), freq_model_enc)
            aec_encoding = aec_encoder.encode_block(dist_arr_data).tobytes()
            encoding_size = len(aec_encoding)
            storage_cost += encoding_size*8
            encoding_size = encoding_size.to_bytes(8, sys.byteorder)
            f.write(encoding_size)
            f.write(aec_encoding)
            
            # Encode the pairwise joint histogram
            # Encode the histogram of histogram (also called fingerprint) in plain text
            data_freq_set = pairwise_hist_list[col_idx].values()
            data_freq_set = DataBlock(data_freq_set)
            freq_of_freq = data_freq_set.get_counts()
            fingerprint_serialized = pickle.dumps(freq_of_freq)
            fingerprint_compressed = gzip.compress(bytes(fingerprint_serialized))
            len_compressed = len(fingerprint_compressed)
            storage_cost += len_compressed*8
            len_compressed =len_compressed.to_bytes(8, sys.byteorder)
            f.write(len_compressed)
            f.write(fingerprint_compressed)
            

            # Encode the second column (frequencies) using AEC
            freq_freq_set = Frequencies(freq_of_freq)
            params = AECParams()
            freq_model_enc = AdaptiveIIDFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
            aec_encoder = ArithmeticEncoder(AECParams(), freq_model_enc)
            aec_encoding = aec_encoder.encode_block(data_freq_set).tobytes()
            encoding_size = len(aec_encoding)
            storage_cost += encoding_size*8
            encoding_size = encoding_size.to_bytes(8, sys.byteorder)
            f.write(encoding_size)
            f.write(aec_encoding) 
            

        f.close()

    # print(joint_entropy_list)
    # print(pairwise_mutual_info)
    return pairwise_mutual_info, storage_cost


# Construct the Chow-Liu tree
def construct_chow_liu_tree(pairwise_mutual_info, storage_cost, num_rows):
    G = nx.Graph()
    # Every vertex in this graph is a feature in the original tabular data
    for x1 in range(num_features):
        G.add_node(x1)
        for x2 in range(x1):
            w = (-pairwise_mutual_info[(x2, x1)])+((1/num_rows)*storage_cost)
            G.add_edge(x2, x1, weight=w)
    chow_liu_tree = nx.minimum_spanning_tree(G)

    # Generate the adjanceny list
    print(chow_liu_tree.edges.data())
    # Print the Chow-Liu tree
    nx.draw(chow_liu_tree, with_labels=True)
    plt.show()

    # Encode the Chow-Liu tree in bytes (plain text) for the decoder


def chow_liu_encoder(data, num_features, num_rows):
    # test_chow_liu_tree()
    # Create the dictionary and ordered data from the read CSV
    dictionary, ordering = create_dict_ordering(data)

    # Encode the dictionary in plain text and write this to the compressed file
    # Encode this using gzip - | num_total_bytes_written_by_gzip | compressed_content_by_gzip |
    encode_dictionary(dictionary)

    # Create and encode the marginal histogram
    enc_size_marg_hist, self_entropy_list, storage_cost1 = create_marginal_hist(ordering)

    # Create and encode the pairwise joint histogram
    mutual_info, storage_cost2 = create_pairwise_joint_hist(ordering, num_features, self_entropy_list, dictionary)

    # Generate the Chow-Liu tree
    storage_cost = storage_cost1 + storage_cost2
    construct_chow_liu_tree(mutual_info, storage_cost, num_rows)

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
# len(support set) | len(encoded_hist_of_hist) | encoded_hist_of_hist | len(encoded_marginal_hist) | encoded_marginal_hist
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
            params = AECParams()
            freq_model_dec = AdaptiveIIDFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
            aec_decoder = ArithmeticDecoder(AECParams(), freq_model_dec)
            aec_decoding, bits_consumed = aec_decoder.decode_block(data_freq_bits)
            
            # Re-create the marginal histogram
            marginal_hist = dict(zip(support_set, aec_decoding.data_list))
            marginal_hist_list.append(marginal_hist)
        
        # print(marginal_hist_list)
        pos_file = f.tell()
        f.close()
    
    return pos_file, marginal_hist_list


# Decode the pairwise joint histogram for all columns
def decode_pairwise_joint_hist(pos_file, n_feat, dict_cols):
    pairwise_hist_list = []
    print("Inside Golomb decoder")
    with open(out_filename, "rb") as f:
        f.seek(pos_file)
        for index1 in range(n_feat):
            for index2 in range((index1+1), n_feat):
                # Decode the support set
                len_support_set = int.from_bytes(f.read(8), sys.byteorder)
                len_encoding = int.from_bytes(f.read(8), sys.byteorder)
                data = f.read(len_encoding)
                dist_freq_bytes = gzip.decompress(data)
                dist_freq = Frequencies(pickle.loads(dist_freq_bytes))

                dist_arr_size = int.from_bytes(f.read(8), sys.byteorder)
                dist_arr = f.read(dist_arr_size)
                dist_arr_bits = BitArray()
                dist_arr_bits.frombytes(dist_arr)
                params = AECParams()
                freq_model_dec = AdaptiveIIDFreqModel(dist_freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_decoder = ArithmeticDecoder(AECParams(), freq_model_dec)
                aec_decoding, bits_consumed = aec_decoder.decode_block(dist_arr_bits)
                decoded_dist_arr = aec_decoding.data_list
                idx_true = np.concatenate(([decoded_dist_arr[0]], decoded_dist_arr[1:])).cumsum()
                idx_true = idx_true.tolist()
                support_set_1d = np.zeros((len_support_set, 1))
                support_set_1d[idx_true] = 1
                support_set_1d = support_set_1d.transpose()[0]
                m = len(dict_cols[index1])
                n = len(dict_cols[index2])
                joint_support_set = support_set_1d.reshape((m, n))
                # print(joint_support_set)

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
                params = AECParams()
                freq_model_dec = AdaptiveIIDFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_decoder = ArithmeticDecoder(AECParams(), freq_model_dec)
                aec_decoding, bits_consumed = aec_decoder.decode_block(data_freq_bits)
                
                # Re-create the pairwise histogram
                non_zeros_idxs = np.nonzero(joint_support_set)
                row_idxs = non_zeros_idxs[0]
                col_idxs = non_zeros_idxs[1]
                support_set_list = list(zip(row_idxs, col_idxs))
                pairwise_hist = dict(zip(support_set_list, aec_decoding.data_list))
                print(pairwise_hist)
                pairwise_hist_list.append(pairwise_hist)

                        
        # print(marginal_hist_list)
        pos_file = f.tell()
        f.close()
    
    return pos_file


def chow_liu_decoder(num_features):
    # Position to track the seeker in the compressed file
    pos_outfile = 0

    # Decode the dictionary from the compressed file
    pos_outfile, dictionary_all_cols = decode_dictionary()

    # Decode the marginal histogram 
    pos_outfile, marginal_hist_list = decode_marginal_hist(pos_outfile, num_features)

    # Decode the pairwise joint histogram
    pos_outfile = decode_pairwise_joint_hist(pos_outfile, num_features, dictionary_all_cols)

   


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
    num_rows = data.shape[0]
    chow_liu_encoder(data, num_features, num_rows)
    
    chow_liu_decoder(num_features)
