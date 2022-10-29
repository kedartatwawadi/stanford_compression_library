from base64 import encode
import code
import pickle
from typing import Callable
from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
import numpy as np
import random
import scipy
from utils.bitarray_utils import BitArray, bitarray_to_uint, get_bit_width, uint_to_bitarray
from external_compressors.pickle_external import PickleEncoder, PickleDecoder
def test_pickle_data_compressor():
    p_enc = PickleEncoder()
    p_dec = PickleDecoder()

    # pickle should work for arbitrary data
    data_list = [3, "alpha", "33.2313241234"]
    encoded_bits = p_enc.encode_block(data_list)
    encoded_bits_and_extra_bits = encoded_bits + BitArray("101111")
    data_list_decoded, num_bits_consumed = p_dec.decode_block(encoded_bits_and_extra_bits)
    assert num_bits_consumed == len(encoded_bits)
    for d1,d2 in zip(data_list, data_list_decoded):
        assert d1 == d2

    data_ordered_dict = {"A": 1.111, "B": 0.3412452, "C": 0.1213441}
    encoded_bits = p_enc.encode_block(data_ordered_dict)
    encoded_bits_and_extra_bits = encoded_bits + BitArray("101111")
    data_dict_decoded, num_bits_consumed = p_dec.decode_block(encoded_bits_and_extra_bits)
    assert num_bits_consumed == len(encoded_bits)

    for d1,d2 in zip(data_ordered_dict, data_dict_decoded):
        assert d1 == d2
        assert data_ordered_dict[d1] == data_dict_decoded[d2]


class UniformScalarQuantizer:
    def __init__(self, quantization_width) -> None:
        self.qw = quantization_width
        assert self.qw > 0
        
    def quantize(self, data_npy: np.array):
        # quantize and convert to int
        return np.array(np.round(data_npy/self.qw), dtype=int)

    def dequantize(self, data_quantized_npy: np.array):
        # dequantize and convert to float
        return np.array(data_quantized_npy*self.qw, dtype=float)

def test_uniform_scalar_quantization():
    for qw in [0.01, 0.1, 0.3]:
        quantizer = UniformScalarQuantizer(qw)

        # test random data
        test_data = np.random.rand(10000)
        quantized_data = quantizer.quantize(test_data)
        dequantized_data = quantizer.dequantize(quantized_data)
        l1_loss = np.linalg.norm((test_data - dequantized_data), ord=1)/len(test_data)

        assert l1_loss <= qw


def l2_loss(v1, v2):
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    d = np.linalg.norm(v1_np-v2_np, ord=2) #l2_norm
    loss = d*d/(v1_np.size) #avg l2 loss
    return loss


class GLVectorLossyEncoderFastMSE(DataEncoder):
    def build_codebook(self, data_npy, dist_thresh=1e-4, max_iter=100):
        codebook,_ = scipy.cluster.vq.kmeans(data_npy, self.num_vectors, iter=max_iter, thresh=dist_thresh, seed=0)
        return codebook


def find_nearest(codebook_npy, data_npy, dist_func):
    distances = [dist_func(c, data_npy) for c in codebook_npy]
    min_ind = np.argmin(distances)
    return min_ind, distances[min_ind]


def build_kmeans_codebook_fast(self, data_npy, dist_thresh=1e-4, max_iter=100):
    codebook,_ = scipy.cluster.vq.kmeans(data_npy, self.num_vectors, iter=max_iter, thresh=dist_thresh, seed=0)
    return codebook

def build_kmeans_codebook(self, data_npy, dist_thresh=1e-4, max_iter=100):
    # define initial tensors to be some of the sample sampling (without replacement)
    data_npy = [np.array(v) for v in data_npy.tolist()]
    codebook = random.sample(data_npy, self.num_vectors)
    codebook = np.stack(codebook, axis=0)

    # for the entire dataset, find the closest centroid
    avg_distortion_arr = []
    for it in range(max_iter): # stopping criteria can be different

        # define intermediate tensors and vals
        counts = np.ones(self.num_vectors)
        centroids = codebook.copy() #current centroid
        sum_distances = 0

        # go over the input data and average
        for data_vector in data_npy:
            min_ind, dist = self.find_nearest(codebook, data_vector, self.distortion_func)
            centroids[min_ind] = (centroids[min_ind]*counts[min_ind] + data_vector)/(counts[min_ind] + 1)
            counts[min_ind] += 1
            sum_distances += dist
        
        avg_dist = sum_distances/len(data_npy)
        codebook = centroids.copy()
        # print(f"it[{it}]: avg_dist: {avg_dist}")
        if it > 0:
            if abs(avg_distortion_arr[-1] - avg_dist) < dist_thresh:
                break
        avg_distortion_arr.append(avg_dist)

    return codebook

class VectorQuantizerEncoder(DataEncoder):
    def __init__(self, codebook_npy, distortion_func) -> None:
        self.codebook = codebook_npy
        self.distortion_func = distortion_func
        self.num_vectors, self.dim = self.codebook.shape
        self.num_vector_bits = self.get_codebook_size_bits(self.num_vectors)

        self.DATA_SIZE_BITS = 32

    @staticmethod
    def get_codebook_size_bits(num_vectors):
        return 1 if (num_vectors == 1) else int(np.ceil(np.log2(num_vectors)))

    def encode_block(self, data_block: DataBlock):
        # assert that data_block size is a multiple of dim
        assert data_block.size % self.dim == 0

        # create vectors
        
        # write codebook to bitstream
        # just use pickle for now
        pickle_enc = PickleEncoder()
        codebook_bits = pickle_enc.encode_block(self.codebook)

        # quantize and convert to int

        # convert input data to npy and reshape to vectors
        data_vectors = np.reshape(np.array(data_block.data_list), [-1, self.dim])
        encoded_bits = uint_to_bitarray(data_vectors.shape[0], self.DATA_SIZE_BITS)
        
        # encode the vectors
        for v in data_vectors:
            min_ind,_ = find_nearest(self.codebook, v, self.distortion_func)
            encoded_bits += uint_to_bitarray(min_ind, self.num_vector_bits)
        return codebook_bits + encoded_bits


class VectorQuantizerDecoder(DataDecoder):
    def __init__(self):
        self.DATA_SIZE_BITS = 32

    def decode_block(self, encoded_bitarray: BitArray):
        # decode the codebook first
        pickle_dec = PickleDecoder()
        self.codebook, num_bits_consumed = pickle_dec.decode_block(encoded_bitarray)

        num_vectors = len(self.codebook)
        num_vector_bits = 1 if (num_vectors == 1) else int(np.ceil(np.log2(num_vectors)))

        # finally decode the data
        decoded_list = []

        # read in data_size
        data_len = bitarray_to_uint(encoded_bitarray[num_bits_consumed:(num_bits_consumed+self.DATA_SIZE_BITS)])
        num_bits_consumed += self.data_size_bits

        # decode vectors
        for _ in range(data_len):
            q = bitarray_to_uint(encoded_bitarray[num_bits_consumed:(num_bits_consumed + num_vector_bits)])
            num_bits_consumed += num_vector_bits
            decoded_list += list(self.codebook[q]) #convert numpy vectors to list for output     
        
        return DataBlock(decoded_list), num_bits_consumed

def test_gl_vector_quantization():
    # generate random uniform data in [0,1]
    DATA_SIZE = 12000
    data = DataBlock(list(np.random.randn(DATA_SIZE)))
    print()
    for num_bits_per_dim in [1]:
        for dim in [1,2,4,8]:
            num_vector_bits = dim*num_bits_per_dim
            num_vectors = 1 << num_vector_bits

            # define enc, dec
            vec_enc = GLVectorLossyEncoder(dim, num_vectors, l2_loss)
            vec_dec = GLVectorLossyDecoder()

            # decode
            encoded_bits = vec_enc.encode_block(data)
            decoded_data,num_bits_consumed = vec_dec.decode_block(encoded_bits)

            assert num_bits_consumed == len(encoded_bits)
            # get stats
            loss = l2_loss(data.data_list, decoded_data.data_list)
            avg_bits = len(encoded_bits)/data.size
            print(f"data_size: {DATA_SIZE}, dim: {dim}, num_vector_bits: {num_vector_bits}, avg_bits: {avg_bits}, loss: {loss}")
    

def test_gl_vector_quantization_fast():
    # generate random uniform data in [0,1]
    DATA_SIZE = 120000
    data = DataBlock(list(np.random.randn(DATA_SIZE)))
    print()
    for num_bits_per_dim in [1]:
        for dim in [1,2,4]:
            num_vector_bits = dim*num_bits_per_dim
            num_vectors = 1 << num_vector_bits

            # define enc, dec
            vec_enc = GLVectorLossyEncoderFast(dim, num_vectors, l2_loss)
            vec_dec = GLVectorLossyDecoder()

            # decode
            encoded_bits = vec_enc.encode_block(data)
            decoded_data,num_bits_consumed = vec_dec.decode_block(encoded_bits)

            assert num_bits_consumed == len(encoded_bits)
            # get stats
            loss = l2_loss(data.data_list, decoded_data.data_list)
            avg_bits = len(encoded_bits)/data.size
            print(f"data_size: {DATA_SIZE}, dim: {dim}, num_vector_bits: {num_vector_bits}, avg_bits: {avg_bits}, loss: {loss}")
    



    


