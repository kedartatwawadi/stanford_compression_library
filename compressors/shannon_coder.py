"""Shannon Coding
Shannon coding is a basic prefix free coding. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of https://en.wikipedia.org/wiki/Shannon–Fano_coding.

This document uses cumulative probability based method for shannon coding as described in the wiki article above.
"""

from bitarray import bitarray
from utils.bitarray_utils import float_to_bitarrays, BitArray, uint_to_bitarray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeEncoder, PrefixFreeDecoder
from core.prob_dist import ProbabilityDist
import math
import numpy as np


class ShannonEncoder(PrefixFreeEncoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist

        # sort the probabilities in descending order
        self.sorted_prob_dist = prob_dist.sorted_prob_list

        # compute a dictionary holding cumulative probabilities
        self.cumulative_prob_dict = self.sorted_prob_dist.cumulative_prob_dict

    def encode_symbol(self, symbol) -> BitArray:
        # compute the mid-point corresponding to the range of the given symbol
        cum_prob = self.cumulative_prob_dict[symbol]

        # compute encode length
        encode_len = math.ceil(self.prob_dist.log_probability(symbol))

        # the encode is the binary representation of the cumulative probability in ascending order,
        # truncated to expected code-word length
        _, code = float_to_bitarrays(cum_prob, encode_len)

        return code


class ShannonDecoder(PrefixFreeDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.sorted_prob_dist = prob_dist.sorted_prob_list
        self.cumulative_prob_dict = self.sorted_prob_dist.cumulative_prob_dict

    def get_codeword(self, symbol) -> BitArray:
        # compute the mid-point corresponding to the range of the given symbol
        cum_prob = self.cumulative_prob_dict[symbol]
        # compute encode length
        encode_len = math.ceil(self.prob_dist.log_probability(symbol))
        # the encoded value is the binary representation of the cumulative probability in ascending order,
        # truncated to expected code-word length
        _, code = float_to_bitarrays(cum_prob, encode_len)
        return code

    def prefix_free_codes(self, prob_dist):
        return {a: self.get_codeword(a) for a in prob_dist.prob_dict.keys()}

    def decode_symbol(self, encoded_bitarray):
        prefix_free_code_dict = self.prefix_free_codes(self.prob_dist)

        # start decoding, stop when the range is fully inside two cumulative intervals
        num_bits_consumed = 0
        undecoded_bits = bitarray()

        while True:
            bit = uint_to_bitarray(encoded_bitarray[num_bits_consumed])
            num_bits_consumed += 1

            undecoded_bits += bit

            for a, c in prefix_free_code_dict.items():
                if undecoded_bits == c:
                    decoded_symbol = a
                    return decoded_symbol, num_bits_consumed


def test_shannon_coding():
    """test if shannon fano elais encoding is lossless"""
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.3, "B": 0.3, "C": 0.4}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
    ]
    for prob_dist in distributions:
        # generate random data
        data_block = get_random_data_block(prob_dist, NUM_SAMPLES, seed=0)

        # create encoder decoder
        encoder = ShannonEncoder(prob_dist)
        decoder = ShannonDecoder(prob_dist)

        # perform compression
        is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
        assert is_lossless, "Lossless compression failed"
