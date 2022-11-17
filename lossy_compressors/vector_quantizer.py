from compressors.fixed_bitwidth_compressor import (
    FixedBitwidthDecoder,
    FixedBitwidthEncoder,
)
from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
import numpy as np
import scipy
from utils.bitarray_utils import BitArray

def mse(v1, v2):
    d = np.linalg.norm(v1 - v2, ord=2)  # l2_norm
    loss = d * d / (v1.size)  # avg l2 loss
    return loss

def find_nearest(codebook_npy, data_vector_npy, dist_func):
    """
    codebook_npy -> [V,D] where V -> num vectors in the codebook, D -> dim of each vector
    data_vector_npy -> [D] sized vector
    """
    distances = [dist_func(c, data_vector_npy) for c in codebook_npy]
    min_ind = np.argmin(distances)
    return min_ind, distances[min_ind]


def build_kmeans_codebook_scipy(data_npy, num_vectors, dim, dist_thresh=1e-4, max_iter=100):
    """
    NOTE: implicitly assumes dist_func is mse
    output codebook -> [num_vectors,dim]
    data_npy -> [N] sized
    """
    data_npy_2d = np.reshape(data_npy, [-1, dim]) #[N/D, D] sizes
    codebook, _ = scipy.cluster.vq.kmeans(
        data_npy_2d, num_vectors, iter=max_iter, thresh=dist_thresh, seed=0
    )
    return codebook


def build_kmeans_codebook(
    data_npy , num_vectors, dim, distortion_func=mse, dist_thresh=1e-4, max_iter=100
):
    """
    NOTE: slow, as we are doing exact nearest neighbor etc.
    output codebook -> [num_vectors,dim]
    data_npy -> [N] sized numpy array, where N is input data
    """
    codebook = np.zeros((num_vectors, dim)) #initialize codebook
    #####################################################
    # ADD DETAILS HERE
    ######################################################
    data_npy_2d = np.reshape(data_npy, [-1, dim])

    # TODO: add code here to set the codebook appropriately
    # using K-means

    #####################################################
    return codebook

class VectorQuantizer:
    def __init__(self, codebook_npy) -> None:
        self.codebook = codebook_npy
        self.num_vectors, self.dim = self.codebook.shape

    def quantize(self, data_npy: np.array, distortion_func: callable):
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
            min_ind, _ = find_nearest(self.codebook, v, distortion_func)
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


class VQEncoder(DataEncoder):
    def __init__(self, codebook_npy, distortion_func) -> None:
        self.distortion_func = distortion_func
        self.vq = VectorQuantizer(codebook_npy)

        # encodes data using a fixed number of bits (based on the alphabet size)
        # also appends data with the codelength
        self.fixed_bitwidth_enc = FixedBitwidthEncoder()

    def encode_block(self, data_npy: np.array):
        assert data_npy.size % self.vq.dim == 0

        # vector quantize data, and encode
        quantized_ind_block = self.vq.quantize(data_npy, self.distortion_func)
        encoded_bits = self.fixed_bitwidth_enc.encode_block(quantized_ind_block)
        return encoded_bits


class VQDecoder(DataDecoder):
    def __init__(self, codebook) -> None:
        self.fixed_bitwidth_dec = FixedBitwidthDecoder()
        self.codebook = codebook

    def decode_block(self, encoded_bitarray: BitArray):
        self.vq = VectorQuantizer(self.codebook)

        # decode the indices
        quantized_ind_block, num_bits_consumed = self.fixed_bitwidth_dec.decode_block(encoded_bitarray)

        # dequantize
        data_npy_decoded = self.vq.dequantize(quantized_ind_block)
        return data_npy_decoded, num_bits_consumed


def _vq_experiment(data_npy, codebook_npy):
    # NOTE: to make this a "true" encoder, we should add the codebook to the bitstream
    # but for simplicity, we won't do that here

    # define enc, dec
    vec_enc = VQEncoder(codebook_npy, distortion_func=mse)
    vec_dec = VQDecoder(codebook_npy)

    # encode/decode
    encoded_bits = vec_enc.encode_block(data_npy)
    decoded_data_npy, num_bits_consumed = vec_dec.decode_block(encoded_bits)
    assert num_bits_consumed == len(encoded_bits)

    # get stats
    avg_mse_distortion = mse(data_npy, decoded_data_npy)
    avg_bits = len(encoded_bits) / (data_npy.size)
    return avg_bits, avg_mse_distortion

def test_vector_quantization_scipy():
    # generate random uniform data in [0,1]
    # NOTE -> slow if we increase DATA_SIZE
    DATA_SIZE = 12000
    RATE = 1 # bit/symbol

    # generate uniform random samples
    data_npy = np.random.randn(DATA_SIZE)

    for dim in [1, 2, 4]:
        num_vectors = 1 << (dim * RATE)

        # get codebook
        codebook_npy = build_kmeans_codebook_scipy(data_npy, num_vectors, dim)
        avg_bits, avg_mse_distortion = _vq_experiment(data_npy, codebook_npy)
        print(
            f"data_size: {DATA_SIZE}, dim: {dim}, num_vectors: {num_vectors}, designed_rate: {RATE:.2f} bits/symbol, total_avg_bits: {avg_bits:.2f} bits/symbol, avg_mse_distortion: {avg_mse_distortion}"
        )  

def test_vector_quantization():
    # generate random uniform data in [0,1]
    # NOTE -> slow if we increase DATA_SIZE
    DATA_SIZE = 12000
    RATE = 1 # bit/symbol
    expected_mse_distortion = {1: 0.36, 2: 0.358, 4: 0.33} #{dim: expected_mse}

    # generate uniform random samples
    data_npy = np.random.randn(DATA_SIZE)

    for dim in [1, 2, 4]:
        num_vectors = 1 << (dim * RATE)
        
        # get codebook
        codebook_npy = build_kmeans_codebook(data_npy, num_vectors, dim)
        avg_bits, avg_mse_distortion = _vq_experiment(data_npy, codebook_npy)
        print(
            f"data_size: {DATA_SIZE}, dim: {dim}, num_vectors: {num_vectors}, designed_rate: {RATE:.2f} bits/symbol, total_avg_bits: {avg_bits:.2f} bits/symbol, avg_mse_distortion: {avg_mse_distortion}"
        )
        assert avg_mse_distortion < (expected_mse_distortion[dim] + 1e-1)

