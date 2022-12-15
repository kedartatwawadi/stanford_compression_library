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
            freq_model_enc = FixedFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
            aec_encoder = ArithmeticEncoder(AECParams(), freq_model_enc)
            aec_encoding = aec_encoder.encode_block(data_freq_set).tobytes()
            encoding_size = len(aec_encoding)
            storage_cost += len(aec_encoding) 
            encoding_size = encoding_size.to_bytes(8, sys.byteorder)
            f.write(encoding_size)
            f.write(aec_encoding)
                 
        
        f.close()
    # print(marginal_hist_list)
    return self_entropy_list, marginal_hist_list, storage_cost

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
    # print(pairwise_columns)

    # Now obtain the pairwise joint histogram for each pairwise combination
    # Also Calculate mutual information for all pairs of columns
    # I(X; Y) = H(X) + H(Y) − H(X, Y)
    # For now, this is the weight that we use for the edges of the Chow-Liu tree
    pairwise_hist_list = {}
    joint_entropy_list = {}
    pairwise_mutual_info = {}
    for column in pairwise_columns:
        (index1, index2) = column
        pairwise_hist = pairwise_columns[column].value_counts(sort=False).to_dict()
        pairwise_hist = OrderedDict(sorted(pairwise_hist.items()))
        pairwise_hist_list[(index1, index2)] = pairwise_hist
        pairwise_prob_dist = Frequencies(pairwise_hist).get_prob_dist()
        joint_entropy_list[(index1, index2)] = pairwise_prob_dist.entropy

        mutual_info = self_entropy_list[index1] + self_entropy_list[index2] - joint_entropy_list[(index1, index2)]
        pairwise_mutual_info[(index1, index2)] = mutual_info

    assert(len(joint_entropy_list) == num_pairs)
    assert(len(pairwise_mutual_info) == num_pairs)
    # print(pairwise_columns)

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
            column_list = pairwise_hist_list[(index1, index2)].keys()
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
            freq_model_enc = FixedFreqModel(dist_freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
            aec_encoder = ArithmeticEncoder(AECParams(), freq_model_enc)
            aec_encoding = aec_encoder.encode_block(dist_arr_data).tobytes()
            encoding_size = len(aec_encoding)
            storage_cost += encoding_size*8
            encoding_size = encoding_size.to_bytes(8, sys.byteorder)
            f.write(encoding_size)
            f.write(aec_encoding)
            
            # Encode the pairwise joint histogram
            # Encode the histogram of histogram (also called fingerprint) in plain text
            data_freq_set = pairwise_hist_list[(index1, index2)].values()
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
            freq_model_enc = FixedFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
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
    # print(pairwise_hist_list)
    return pairwise_mutual_info, pairwise_hist_list, pairwise_columns, storage_cost


# Construct the Chow-Liu tree
def construct_chow_liu_tree(pairwise_mutual_info, storage_cost, num_rows, n_feat):
    G = nx.Graph()
    # Every vertex in this graph is a feature in the original tabular data
    for x1 in range(n_feat):
        G.add_node(x1)
        for x2 in range(x1):
            w = (-pairwise_mutual_info[(x2, x1)])+((1/num_rows)*storage_cost)
            G.add_edge(x2, x1, weight=w)
    chow_liu_tree = nx.minimum_spanning_tree(G)

    # Generate the adjanceny list
    print("Chow-Liu tree Edges: ", chow_liu_tree.edges.data())
    # Print the Chow-Liu tree
    nx.draw(chow_liu_tree, with_labels=True)
    plt.savefig('chow-liu.png')
    # plt.show()

    # Encode the Chow-Liu tree in bytes (plain text) for the decoder

    return chow_liu_tree


# Encode the ordered data representing the tabular CSV
def encode_data(dictionary, ordered_data, chow_liu_tree, marginal_hist, pairwise_hist, pairwise_columns):
    # print("Inside encoder@@@@@@@@@@@@@")
    # Get the edges of the chow_liu tree in a BFS manner 
    data_edges_bfs = list(nx.edge_bfs(chow_liu_tree))

    # From the pairwise histogram, construct a table of conditional probabilities
    # P(Y | X) for all the source(X)-edge(Y) combinations
    # This will be a list of dictionary of list of dictionaries
    # {(X1,Y1): [{}],  (X1, Y2):[].............}
    # The value of (X1,Y1) here is a list where each element represents frequencies 
    # for conditioning on some X=a
    conditional_prob_all = {}
    for edge in data_edges_bfs:
        col1, col2 = edge
        pairwise_freq = pairwise_hist[(col1, col2)]
        conditional_prob_list = []
        for x_val in dictionary[col1].keys():
            freqs_y_given_x = {}
            for y_val in dictionary[col2].keys():
                # compute the conditional probability P(Y=a|X=i)
                if((x_val, y_val) not in pairwise_freq):
                    freqs_y_given_x[y_val] = 0
                else:
                    freqs_y_given_x[y_val] = pairwise_freq[(x_val, y_val)]
            
            assert(len(freqs_y_given_x)==len(dictionary[col2]))
            conditional_prob_list.append(freqs_y_given_x)        
        
        assert(len(conditional_prob_list)==len(dictionary[col1]))
        conditional_prob_all[(col1, col2)] = conditional_prob_list       

    # print(conditional_prob_all)
    # Maintain a set of already encoded nodes
    encoded_nodes = set()

    # Iterate over the list of edges - every child has a single parent
    # Encode the very first node using it's marginal histogram
    # Then encode it's child nodes using the conditional probability (joint histogram)
    # Use Arithmetic encoding for encoding
    with open(out_filename, "ab") as f:
        for edge in data_edges_bfs:
            source, dest = edge     
            # Encode the source using it's marginal distribution
            # Encode the whole source as a block
            if(source not in encoded_nodes):
                freq = Frequencies(marginal_hist[source])
                params = AECParams()
                freq_model = FixedFreqModel(freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_enc = ArithmeticEncoder(AECParams(), freq_model)
                data = DataBlock(ordered_data[source])
                encoding = aec_enc.encode_block(data).tobytes()
                encoding_size = len(encoding).to_bytes(8, sys.byteorder)
                f.write(encoding_size)
                f.write(encoding)
                encoded_nodes.add(source)
            
            # Encode the destination using conditional probabilities
            # This is a encoding is broken over multiple blocks as we feed in frequencies
            # that are conditioned on what the alphabet in source is
            # Encoding is in the order of dictionary for the source data
            conditional_prob = conditional_prob_all[(source, dest)]
            # print(conditional_prob)
            for x_val in dictionary[source].keys():
                # print("Conditioning for:", x_val)
                freq = Frequencies(conditional_prob[x_val])
                # print(freq)
                freq_model = FixedFreqModel(freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_enc = ArithmeticEncoder(AECParams(), freq_model)
                # Select all the rows in col1 with X=x_val
                data = DataBlock(ordered_data[dest].loc[ordered_data[source]==x_val])
                encoding = aec_enc.encode_block(data).tobytes()
                encoding_size = len(encoding).to_bytes(8, sys.byteorder)
                f.write(encoding_size)
                f.write(encoding)
            
            encoded_nodes.add(dest)
            
        # For all the nodes that remain to be encoded - find their predecessor and encode them
        nodes = nx.nodes(chow_liu_tree)
        assert(len(encoded_nodes)==len(nodes))
        f.close()
            


def chow_liu_encoder(data, num_features, num_rows):
    # test_chow_liu_tree()
    # Create the dictionary and ordered data from the read CSV
    # print("Data")
    # print(data)
    dictionary, ordering = create_dict_ordering(data)
    # print("Dictionary")
    # print(dictionary)
    # print("Ordering")
    # print(ordering)
    print("-----------Created the dictionary and ordering for the dataset------------")

    # Encode the dictionary in plain text and write this to the compressed file
    # Encode this using gzip - | num_total_bytes_written_by_gzip | compressed_content_by_gzip |
    encode_dictionary(dictionary)
    print("-----------Encoded the dictionary------------")

    # Create and encode the marginal histogram
    self_entropy_list, marginal_hist, storage_cost1 = create_marginal_hist(ordering)
    # print("Marginal Histogram: ", marginal_hist)
    print("-----------Created & encoded the marginal histogram for the dataset------------")

    # Create and encode the pairwise joint histogram
    mutual_info, pairwise_hist, pairwise_columns, storage_cost2 = create_pairwise_joint_hist(ordering, num_features, self_entropy_list, dictionary)
    # print("Pairwise Histogram: ", pairwise_hist)
    print("-----------Created & encoded the pairwise joint histogram for the dataset------------")

    # Generate the Chow-Liu tree
    storage_cost = storage_cost1 + storage_cost2
    chow_liu_tree = construct_chow_liu_tree(mutual_info, storage_cost, num_rows, num_features)
    print("-----------Constructed the Chow-Liu tree------------")

    # Perform pairwise encoding of columns in the dataset
    encode_data(dictionary, ordering, chow_liu_tree, marginal_hist, pairwise_hist, pairwise_columns)
    print("-----------Finished Encoding!------------")

    # print(ordering)

    return chow_liu_tree, ordering



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
            freq_model_dec = FixedFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
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
    pairwise_hist_list = {}
    print("Inside Decoder")
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
                freq_model_dec = FixedFreqModel(dist_freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
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
                freq_model_dec = FixedFreqModel(freq_freq_set, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_decoder = ArithmeticDecoder(AECParams(), freq_model_dec)
                aec_decoding, bits_consumed = aec_decoder.decode_block(data_freq_bits)
                
                # Re-create the pairwise histogram
                non_zeros_idxs = np.nonzero(joint_support_set)
                row_idxs = non_zeros_idxs[0]
                col_idxs = non_zeros_idxs[1]
                support_set_list = list(zip(row_idxs, col_idxs))
                pairwise_hist = dict(zip(support_set_list, aec_decoding.data_list))
                pairwise_hist_list[(index1, index2)] = pairwise_hist

                        
        # print(pairwise_hist_list)
        pos_file = f.tell()
        f.close()
    
    return pos_file, pairwise_hist_list


def decode_data(pos_file, n_feat, dict_cols, marginal_hist, pairwise_hist, chow_liu_tree):
    # Get the edges of the chow_liu tree in a BFS manner 
    data_edges_bfs = list(nx.edge_bfs(chow_liu_tree))
    ordered_data = pd.DataFrame(columns=list(range(n_feat)))

    # From the pairwise histogram, construct a table of conditional probabilities
    # P(Y | X) for all the source(X)-edge(Y) combinations
    # This will be a list of dictionary of list of dictionaries
    # {(X1,Y1): [{}],  (X1, Y2):[].............}
    # The value of (X1,Y1) here is a list where each element represents frequencies 
    # for conditioning on some X=a
    conditional_prob_all = {}
    for edge in data_edges_bfs:
        col1, col2 = edge
        pairwise_freq = pairwise_hist[(col1, col2)]
        conditional_prob_list = []
        for x_val in dict_cols[col1].keys():
            freqs_y_given_x = {}
            for y_val in dict_cols[col2].keys():
                # compute the conditional probability P(Y=a|X=i)
                if((x_val, y_val) not in pairwise_freq):
                    freqs_y_given_x[y_val] = 0
                else:
                    freqs_y_given_x[y_val] = pairwise_freq[(x_val, y_val)]
            
            assert(len(freqs_y_given_x)==len(dict_cols[col2]))
            conditional_prob_list.append(freqs_y_given_x)        
        
        assert(len(conditional_prob_list)==len(dict_cols[col1]))
        conditional_prob_all[(col1, col2)] = conditional_prob_list       

    # print(conditional_prob_all)

    # Maintain a set of already encoded nodes
    decoded_nodes = set()
    with open(out_filename, "rb") as f:
        f.seek(pos_file)
        for edge in data_edges_bfs:
            source, dest = edge
            # Decode the source using it's marginal distribution
            # Decode the whole source as a block        
            if(source not in decoded_nodes):
                len_data = int.from_bytes(f.read(8), sys.byteorder)
                data_bytes = f.read(len_data)
                data_bits = BitArray()
                data_bits.frombytes(data_bytes)
                data_freq = Frequencies(marginal_hist[source])
                params = AECParams()
                freq_model_dec = FixedFreqModel(data_freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_decoder = ArithmeticDecoder(AECParams(), freq_model_dec)
                aec_decoding, bits_consumed = aec_decoder.decode_block(data_bits)
                ordered_data[source] = aec_decoding.data_list
                decoded_nodes.add(source)
                # print("Decoded node: ", dest)
                # print("Len:, Decoding bytestream: ", len_data, data_bytes)
                # print("Frequency for decoding: ", pairwise_hist[(source, dest)])
                # print("Decoded data: ")
                # print(aec_decoding.data_list)
            
            # Decode the destination using conditional probabilities
            # To form a dest column - we need to decode in multiple passes
            # Decoding is in the order of dictionary for the source data
            assert(source in decoded_nodes)
            conditional_prob = conditional_prob_all[(source, dest)]
            for x_val in dict_cols[source].keys():
                # print("Conditioning for:", x_val)
                len_data = int.from_bytes(f.read(8), sys.byteorder)
                data_bytes = f.read(len_data)
                data_bits = BitArray()
                data_bits.frombytes(data_bytes)
                freq = Frequencies(conditional_prob[x_val])
                params = AECParams()
                freq_model = FixedFreqModel(freq, max_allowed_total_freq=params.MAX_ALLOWED_TOTAL_FREQ)
                aec_decoder = ArithmeticDecoder(AECParams(), freq_model)
                aec_decoding, bits_consumed = aec_decoder.decode_block(data_bits)
                # Select all the rows in col1 with X=x_val, and change the values in col2 to decoded values
                ordered_data[dest].loc[ordered_data[source]==x_val] = aec_decoding.data_list

            decoded_nodes.add(dest)
        
        # For all the nodes that remain to be encoded - find their predecessor and encode them
        nodes = nx.nodes(chow_liu_tree)
        assert(len(decoded_nodes)==len(nodes))
        f.close()
    
    return ordered_data


def chow_liu_decoder(num_features, chow_liu_tree):
    # Position to track the seeker in the compressed file
    pos_outfile = 0

    # Decode the dictionary from the compressed file
    pos_outfile, dictionary_all_cols = decode_dictionary()

    # Decode the marginal histogram 
    pos_outfile, marginal_hist_list = decode_marginal_hist(pos_outfile, num_features)
    # print(marginal_hist_list)

    # Decode the pairwise joint histogram
    pos_outfile, pairwise_hist_list = decode_pairwise_joint_hist(pos_outfile, num_features, dictionary_all_cols)

    ordered_data = decode_data(pos_outfile, num_features, dictionary_all_cols, marginal_hist_list, pairwise_hist_list, chow_liu_tree)
    return ordered_data
   


if __name__ == "__main__":
    # Read the CSV file for our tabular database
    filename = sys.argv[1]
    filename = os.path.abspath(filename)
    # Skip the header row while reading the dataframe
    data = pd.read_csv(filename, skiprows=1, header=None)

    out_filename = filename + ".compressed"
    decompressed_file = filename + ".decompressed.csv"
    if os.path.exists(out_filename):
        os.remove(out_filename)
    
    num_features = len(data.columns)
    num_rows = data.shape[0]
    chow_liu_tree, ordered_data_encoder = chow_liu_encoder(data, num_features, num_rows)
    
    ordered_data_decoder = chow_liu_decoder(num_features, chow_liu_tree)

    print(ordered_data_encoder)
    if os.path.exists(os.path.abspath("ordered_encoder.csv")):
        os.remove("ordered_encoder.csv")
    ordered_data_encoder.to_csv("ordered_encoder.csv")

    print(ordered_data_decoder)
    if os.path.exists(os.path.abspath("ordered_decoder.csv")):
        os.remove("ordered_decoder.csv")
    ordered_data_decoder.to_csv("ordered_decoder.csv")