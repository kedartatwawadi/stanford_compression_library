from typing import List
from compressors.fixed_bitwidth_compressor import (
    FixedBitwidthDecoder,
    FixedBitwidthEncoder,
    get_alphabet_fixed_bitwidth,
)
from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
import numpy as np
import random
import scipy
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from external_compressors.pickle_external import PickleEncoder, PickleDecoder


def mse(v1, v2):
    d = np.linalg.norm(v1 - v2, ord=2)  # l2_norm
    loss = d * d / (v1.size)  # avg l2 loss
    return loss


def find_nearest(codebook_npy, data_npy, dist_func):
    """
    codebook_npy -> [V,D] where V -> num vectors in the codebook, D -> dim of each vector
    data_npy -> [D] sized vector
    """
    distances = [dist_func(c, data_npy) for c in codebook_npy]
    min_ind = np.argmin(distances)
    return min_ind, distances[min_ind]


def build_kmeans_codebook_fast(data_npy, num_vectors, dist_thresh=1e-4, max_iter=100):
    """
    NOTE: implicitly assumes dist_func is mse
    output codebook -> [num_vectors,dim]
    data_npy -> [N/D, D] sizes, where N is the size of data
    """
    codebook, _ = scipy.cluster.vq.kmeans(
        data_npy, num_vectors, iter=max_iter, thresh=dist_thresh, seed=0
    )
    return codebook


def build_kmeans_codebook(
    data_npy, num_vectors, distortion_func=mse, dist_thresh=1e-4, max_iter=100
):
    """
    NOTE: slow, as we are doing exact nearest neighbor etc.
    output codebook -> [num_vectors,dim]
    data_npy -> [N/D, D] sizes, where N is the size of data
    """

    # define initial tensors to be samples from existing data (without replacement)
    data_npy = [np.array(v) for v in data_npy.tolist()]
    codebook = random.sample(data_npy, num_vectors)
    codebook = np.stack(codebook, axis=0)

    ## for the entire dataset, find the closest centroid
    # repeat until convergence
    avg_distortion_arr = []
    for it in range(max_iter):
        # define intermediate tensors and vals
        counts = np.ones(num_vectors)
        centroids = codebook.copy()  # current centroid
        sum_distances = 0

        # go over the input data and average
        for data_vector in data_npy:
            min_ind, dist = find_nearest(codebook, data_vector, distortion_func)
            centroids[min_ind] = (centroids[min_ind] * counts[min_ind] + data_vector) / (
                counts[min_ind] + 1
            )
            counts[min_ind] += 1
            sum_distances += dist

        avg_dist = sum_distances / len(data_npy)
        codebook = centroids.copy()
        # print(f"it[{it}]: avg_dist: {avg_dist}")
        if it > 0:
            if abs(avg_distortion_arr[-1] - avg_dist) < dist_thresh:
                break
        avg_distortion_arr.append(avg_dist)

    return codebook


class VectorQuantizer:
    def __init__(self, codebook_npy, distortion_func) -> None:
        self.codebook = codebook_npy
        self.distortion_func = distortion_func
        self.num_vectors, self.dim = self.codebook.shape

    def quantize(self, data_npy: np.array):
        """
        data_npy -> [N] sized flattened input
        """
        # assert that data_block size is a multiple of dim
        assert len(data_npy.shape) == 1  # np.array
        assert data_npy.size % self.dim == 0
        data_vectors = np.reshape(data_npy, [-1, self.dim])

        # encode the vectors
        quantized_ind_list = []
        for v in data_vectors:
            min_ind, _ = find_nearest(self.codebook, v, self.distortion_func)
            quantized_ind_list.append(min_ind)
        return DataBlock(quantized_ind_list)

    def dequantize(self, quantized_ind_block: DataBlock):
        """
        takes in indices and returns flattened np data
        """
        # dequantize data vectors dim x num_vectors
        dequantized_data = self.codebook[quantized_ind_block.data_list]
        dequantized_data_flat = dequantized_data.flatten()
        return dequantized_data_flat


class VectorQuantizerEncoder(DataEncoder):
    def __init__(self, codebook_npy, distortion_func) -> None:
        self.vq = VectorQuantizer(codebook_npy, distortion_func)
        self.fixed_bitwidth_enc = FixedBitwidthEncoder()

    def encode_block(self, data_npy):
        ## TODO: @tpulkit: add padding if needed to data_npy
        assert data_npy.size % self.vq.dim == 0

        # vector quantize data, and encode
        quantized_ind_block = self.vq.quantize(data_npy)
        vq_bits = self.fixed_bitwidth_enc.encode_block(quantized_ind_block)

        # write codebook to bitstream
        # just use pickle for now
        pickle_enc = PickleEncoder()
        codebook_bits = pickle_enc.encode_block(self.vq.codebook)

        # the final output is codebook bits + vq_bits
        encoded_bits = codebook_bits + vq_bits
        return encoded_bits


class VectorQuantizerDecoder(DataDecoder):
    def __init__(self) -> None:
        self.fixed_bitwidth_dec = FixedBitwidthDecoder()

    def decode_block(self, encoded_bitarray: BitArray):
        # decode the codebook first
        pickle_dec = PickleDecoder()
        codebook, num_bits_consumed = pickle_dec.decode_block(encoded_bitarray)
        self.vq = VectorQuantizer(
            codebook, distortion_func=None
        )  # dist func not used during dequant

        # decode the indices
        quantized_ind_block, num_bits_ind = self.fixed_bitwidth_dec.decode_block(
            encoded_bitarray[num_bits_consumed:]
        )
        num_bits_consumed += num_bits_ind

        # dequantize
        data_npy_decoded = self.vq.dequantize(quantized_ind_block)
        return data_npy_decoded, num_bits_consumed


def test_vector_quantization():
    # generate random uniform data in [0,1]
    # NOTE -> slow if we increase DATA_SIZE
    DATA_SIZE = 12000
    data_npy = np.random.randn(DATA_SIZE)
    print()
    for num_bits_per_dim in [1]:
        for dim in [1, 2]:
            num_vector_bits = dim * num_bits_per_dim
            num_vectors = 1 << num_vector_bits

            # define enc, dec
            data_vectors = np.reshape(data_npy, [-1, dim])
            codebook_npy = build_kmeans_codebook(data_vectors, num_vectors)
            vec_enc = VectorQuantizerEncoder(codebook_npy, distortion_func=mse)
            vec_dec = VectorQuantizerDecoder()

            # decode
            encoded_bits = vec_enc.encode_block(data_npy)
            decoded_data_npy, num_bits_consumed = vec_dec.decode_block(encoded_bits)

            assert num_bits_consumed == len(encoded_bits)
            # get stats
            loss = mse(data_npy, decoded_data_npy)
            avg_bits = len(encoded_bits) / (data_npy.size)
            print(
                f"data_size: {DATA_SIZE}, dim: {dim}, num_vector_bits: {num_vector_bits}, data_bits: {num_vector_bits/dim}, total_avg_bits: {avg_bits}, loss: {loss}"
            )


def test_vector_quantization_fast():
    # TODO: this test is very similar to the other one, refactor later
    # keeping this way for easy experimentation for now
    # generate random uniform data in [0,1]
    DATA_SIZE = 120000
    data_npy = np.random.randn(DATA_SIZE)
    print()
    for num_bits_per_dim in [1]:
        for dim in [1, 2, 4]:
            num_vector_bits = dim * num_bits_per_dim
            num_vectors = 1 << num_vector_bits

            # define enc, dec
            data_vectors = np.reshape(data_npy, [-1, dim])
            codebook_npy = build_kmeans_codebook_fast(data_vectors, num_vectors)
            vec_enc = VectorQuantizerEncoder(codebook_npy, distortion_func=mse)
            vec_dec = VectorQuantizerDecoder()

            # decode
            encoded_bits = vec_enc.encode_block(data_npy)
            decoded_data_npy, num_bits_consumed = vec_dec.decode_block(encoded_bits)

            assert num_bits_consumed == len(encoded_bits)
            # get stats
            loss = mse(data_npy, decoded_data_npy)
            avg_bits = len(encoded_bits) / (data_npy.size)
            print(
                f"data_size: {DATA_SIZE}, dim: {dim}, num_vector_bits: {num_vector_bits}, data_bits: {num_vector_bits/dim}, total_avg_bits: {avg_bits}, mse_loss: {loss}"
            )
