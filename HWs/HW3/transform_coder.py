import numpy as np
from vector_quantizer import VQEncoder, VQDecoder, build_kmeans_codebook_scipy, mse
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from core.data_encoder_decoder import DataDecoder, DataEncoder

class LinearTransform:
    def __init__(self, fwd_matrix: np.array):
        """
        self.fwd_matrix -> [D,D]
        """
        self.fwd_matrix = fwd_matrix
        self.dim = self.fwd_matrix.shape[0]
        self.inv_matrix = np.linalg.inv(self.fwd_matrix)
    
    def forward(self, data_npy):
        """
        Takes in as input data_npy of size [N], and outputs a np.array of shape
        [N/D, D]
        NOTE: the transformed input is kept in this 2D fashion ([N/D, D]), as that is what is 
        the most useful
        """
        data_2d = np.reshape(data_npy, [-1, self.dim])
        data_transformed_2d = np.matmul(data_2d, self.fwd_matrix)
        return data_transformed_2d

    def inverse(self, data_coeff_2d: np.array):
        """
        Takes in as input data_coeff_2d of size [N/D, D], and outputs a np.array of shape
        [N]
        """
        data_inv_2d = np.matmul(data_coeff_2d, self.inv_matrix) 
        data_inv = np.reshape(data_inv_2d, [-1])
        return data_inv


class TransformVQEncoder(DataEncoder):
    def __init__(self, transform: LinearTransform, codebooks_per_coeff: list) -> None:
        self.codebooks_per_coeff = codebooks_per_coeff
        self.transform = transform

    def encode_block(self, data_npy: np.array):
        # transform the input data of shape N
        # data_coeffs_2d shape: [N/D, D]
        data_coeff_2d = self.transform.forward(data_npy)

        # initialize the output bitrarray
        encoded_bitarray = BitArray("")

        # Loop over each coefficient id, and encode it to the bitstream
        for coeff_id in range(self.transform.dim):
            codebook = self.codebooks_per_coeff[coeff_id] #codebook corresp to the coeff
            coeff_data = data_coeff_2d[:,coeff_id] #data corresp to the coeff

            if codebook is not None:
                # Perform VQ using the codebook provided for that coeff
                vq_enc = VQEncoder(codebook)
                encoded_bitarray += vq_enc.encode_block(coeff_data)
            else:
                # if the codebook is None, we only write the len(coeff_data)
                # using 32 bits. 
                # (so that the decoder can convert these to zeros of len(coeff_data))
                encoded_bitarray += uint_to_bitarray(len(coeff_data), bit_width=32)

        return encoded_bitarray

class TransformVQDecoder(DataDecoder):
    def __init__(self, transform: LinearTransform, codebooks_per_coeff: list) -> None:
        self.transform = transform 
        self.codebooks_per_coeff = codebooks_per_coeff

    def decode_block(self, encoded_bitarray: BitArray):
        #####################################################################
        # COMPLETE THE DECODE BLOCK FUNCTION
        # NOTE: You may find it useful to carefully study the TransformVQEncoder
        # code, and understand what is happening there
        # NOTE: please add inline comments (more comments never hurt!)
        #####################################################################
        
        # return decoded_data, num_bits_consumed
        pass
        #####################################################################

def _transform_coding_experiment(data_npy, transform: LinearTransform, vq_dim: int, num_vectors_per_coeff: list):
    
    # Build the codebooks used for vector quantization of individual components
    # NOTE: to make this a "true" encoder, we should add the codebook to the bitstream
    # but for simplicity, we won't do that here
    data_coeff_2d = transform.forward(data_npy)
    codebooks_per_coeffs = []
    for coeff_id in range(transform.dim):
        num_vectors_coeff = num_vectors_per_coeff[coeff_id]
        coeff_data = data_coeff_2d[:,coeff_id]
        if num_vectors_coeff == 1:
            # assumed to be zeros, so the codebook is not saved
            codebook = None
        else:
            codebook = build_kmeans_codebook_scipy(coeff_data,num_vectors_coeff, vq_dim)
        codebooks_per_coeffs.append(codebook)

    # encode
    tvq_enc = TransformVQEncoder(transform, codebooks_per_coeffs)
    tvq_dec = TransformVQDecoder(transform, codebooks_per_coeffs)

    # decode
    encoded_bits = tvq_enc.encode_block(data_npy)
    decoded_data_npy, num_bits_consumed = tvq_dec.decode_block(encoded_bits)
    assert num_bits_consumed == len(encoded_bits)

    # get stats
    avg_mse_distortion = mse(data_npy, decoded_data_npy)
    avg_bits = len(encoded_bits) / (data_npy.size)
    return avg_bits, avg_mse_distortion


def generate_gauss_markov_samples(num_samples, rho, sigma_0=1, sigma=1):
    """
    num_samples -> number of samples to generate
    rho -> correlation between two consecutive samples
    sigma_0 -> std of the initial gaussian
    sigma -> std of the noise term

    returns -> a list of samples from a gaussian markov process such that
    x[n] = rho * x[n-1] + sqrt(1 - rho^2) * z[n]
    where z[n] is a gaussian random variable with mean 0 and std sigma
    x[0] is a gaussian random variable with mean 0 and std sigma_0
    """
    samples = np.zeros(num_samples)
    samples[0] = np.random.normal(0, sigma_0)
    for i in range(1, num_samples):
        samples[i] = rho * samples[i - 1] + np.sqrt((1 - rho**2)) * np.random.normal(0, sigma)
    return samples

def test_transform_vq():
    # NOTE -> slow if we increase DATA_SIZE
    DATA_SIZE = 12000
    RATE = 1 # bit/symbol
    VQ_DIM = 2
    RHO = 0.99
    
    # try a simple transformation matrix
    fwd_matrix = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    transform = LinearTransform(fwd_matrix=fwd_matrix)

    # your mse distortion should be close to this
    expected_avg_mse = 0.021

    # generate random samples
    data_npy = generate_gauss_markov_samples(DATA_SIZE, RHO, sigma_0=1)
    rate_per_coeff = np.array([2, 0]) #per coeff
    num_vectors_per_coeff = np.power(2, rate_per_coeff*VQ_DIM)

    # get codebook
    avg_bits, avg_mse_distortion = _transform_coding_experiment(data_npy, transform, VQ_DIM, num_vectors_per_coeff)
    print(
        f"data_size: {DATA_SIZE}, rate_per_coeff: {rate_per_coeff}, "
        f"num_vectors_per_coeff: {num_vectors_per_coeff}, RATE: {RATE:.2f} bits/symbol, "
        f"total_avg_bits: {avg_bits:.2f} bits/symbol, avg_mse_distortion: {avg_mse_distortion}"
    )

    assert avg_mse_distortion < expected_avg_mse + 1e-2