from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder
from scl.compressors.probability_models import FixedFreqModel
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies, ProbabilityDist
import numpy as np
from scl.utils.bitarray_utils import float_to_bitarrays, uint_to_bitarray


def l1_distance_prob_dist(p1: ProbabilityDist, p2: ProbabilityDist):
    """
    compute the L1 distance between two ProbabilityDist
    """
    combined_alphabet = p1.alphabet + p2.alphabet
    l1_dist = 0
    for a in combined_alphabet:
        p1_val = p1.prob_dict.get(a, 0.0)  # returns 0.0 if not found
        p2_val = p2.prob_dict.get(a, 0.0)  # returns 0.0 if not found
        l1_dist += abs(p1_val - p2_val)
    return l1_dist


def generate_samples_vanilla(freqs: Frequencies, data_size):
    """
    Generate data samples with the given frequencies from uniform distribution [0, 1) using the basic approach
    :param freqs: frequencies of symbols (see Frequencies class)
    :param data_size: number of samples to generate
    :return: DataBlock object with generated samples
    """
    prob_dist = freqs.get_prob_dist()

    # some lists which might be useful
    symbol_list = list(prob_dist.cumulative_prob_dict.keys())
    cumul_list = list(prob_dist.cumulative_prob_dict.values())
    cumul_list.append(1.0)

    generated_samples_list = []  # <- holds generated samples
    for _ in range(data_size):
        # sample a uniform random variable in [0, 1)
        u = np.random.rand()

        ###############################################
        # ADD CODE HERE
        raise NotImplementedError("You need to implement this part")
        ###############################################

    return DataBlock(generated_samples_list)


def generate_samples_aec(freq_initial, data_size):
    """
    Generate data samples with the given frequencies from uniform distribution [0, 1) using the arithmetic entropy
    coding (AEC) approach
    :param freq_initial: frequencies of symbols (see Frequencies class)
    :param data_size: number of samples to generate
    :return: DataBlock object with generated samples obtained from AEC approach
    """
    # generate random variable from uniform distribution [0, 1)
    u = np.random.rand()

    # the bits are the truncation of u to 32 bits
    # NOTE: due to finite arithmetic, we can't go beyond 32, so we truncate u is at max 32 bits
    _, bits = float_to_bitarrays(u, 32)

    # As we are "faking" the Arithmetic encoding, we need to add
    # size of input data to the beginning of the data stream so that we can use our AEC encoder-decoder
    aec_params = AECParams()
    encoded_bitarray = uint_to_bitarray(data_size, aec_params.DATA_BLOCK_SIZE_BITS)
    encoded_bitarray += bits

    # decode the encoded_bitarray using the Arithmetic coder
    freq_model = FixedFreqModel(freq_initial, max_allowed_total_freq=aec_params.DATA_BLOCK_SIZE_BITS)
    aec_decoder = ArithmeticDecoder(aec_params, freq_model)
    data_block_decoded, _ = aec_decoder.decode_block(encoded_bitarray)

    return data_block_decoded


def test_vanilla_generator():
    DATA_SIZE = 10000
    # L1 error threshold between real and empirical distribution
    p_thr = 0.05

    # Frequency distribution of symbols in data
    freq = Frequencies({"A": 3, "B": 4, "C": 1})
    data_block = generate_samples_vanilla(freq, DATA_SIZE)

    # get the source probability distribution from frequencies
    real_dist = freq.get_prob_dist()

    # get the empirical distribution from the generated data
    empirical_dist = data_block.get_empirical_distribution()

    print(f"L1 distance between real and empirical distribution is {l1_distance_prob_dist(real_dist, empirical_dist)}")
    assert l1_distance_prob_dist(real_dist, empirical_dist) < p_thr, "L1 distance between real " \
                                                                     "and empirical distribution is too large"


def test_AEC_generator():
    DATA_SIZE = 10000
    # L1 error threshold between real and empirical distribution
    p_thr = 0.05

    # Frequency distribution of symbols in data
    freq = Frequencies({"A": 3, "B": 4, "C": 1})
    data_block = generate_samples_aec(freq, DATA_SIZE)

    # get the source probability distribution from frequencies
    real_dist = freq.get_prob_dist()

    # get the empirical distribution from the generated data
    empirical_dist = data_block.get_empirical_distribution()

    print(f"L1 distance between real and empirical distribution is {l1_distance_prob_dist(real_dist, empirical_dist)}")
    assert l1_distance_prob_dist(real_dist, empirical_dist) < p_thr, "L1 distance between real " \
                                                                     "and empirical distribution is too large"
