"""Shannon Fano Elias Coding
Shannon Fano Elias coding is a prefix free coding, and is a precursor to Arithmetic coding

Encoding can be seen from here: https://en.wikipedia.org/wiki/Shannon%E2%80%93Fano%E2%80%93Elias_coding
NOTE: As Shannon-Fano-Elias is a prefix-free code, one can compute the codewords once and use that for encoding.
The decoding can also employ the prefix-free tree. We however have implemented both encoding/decoding in a way 
which motivates Arithmetic coding 

"""

from scl.utils.bitarray_utils import float_to_bitarrays, BitArray
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.compressors.prefix_free_compressors import (
    PrefixFreeEncoder,
    PrefixFreeDecoder,
)
from scl.core.prob_dist import ProbabilityDist
import math
import numpy as np


class ShannonFanoEliasEncoder(PrefixFreeEncoder):
    def __init__(self, prob_dist: ProbabilityDist):

        # FIXME: note that there could be issues in computing the
        # if the probability values are too small. We add a check
        self.prob_dist = prob_dist

        # compute a dictionary holding cumulative probabilities
        self.cumulative_prob_dict = self.prob_dist.cumulative_prob_dict

    def encode_symbol(self, symbol) -> BitArray:
        # compute the mid-point corresponding to the range of the given symbol
        cum_prob = self.cumulative_prob_dict[symbol]
        prob = self.prob_dist.probability(symbol)
        F = cum_prob + prob / 2

        # compute encode length
        encode_len = math.ceil(self.prob_dist.neg_log_probability(symbol)) + 1

        # the encode is the binarry representation of the mid-point of the range
        _, code = float_to_bitarrays(F, encode_len)
        return code


class ShannonFanoEliasDecoder(PrefixFreeDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.cumulative_prob_dict = self.prob_dist.cumulative_prob_dict

    def decode_symbol(self, encoded_bitarray):

        # initialize the start/end range
        range_start = 0.0
        range_end = 1.0

        # start decoding, stop when the range is fully inside two cumulative intervals
        num_bits_consumed = 0
        while True:
            bit = encoded_bitarray[num_bits_consumed]
            num_bits_consumed += 1

            # adjust the ranges based on the bit retrieved
            if bit == 0:
                range_end = (range_start + range_end) / 2
            else:
                range_start = (range_start + range_end) / 2

            # check if range_end and range_start are both in the same bucket
            cum_prob_list = list(self.cumulative_prob_dict.values())
            start_bin = np.searchsorted(cum_prob_list, range_start, side="right")
            end_bin = np.searchsorted(cum_prob_list, range_end, side="left")

            if start_bin == end_bin:
                decoded_symbol = self.prob_dist.alphabet[start_bin - 1]

                # FIXME: The recomputing of num_bits_consumed seems necessary, as there is a possibility
                # that the decoder is able to infer the correct symbol by reading less than ceil(-log(prob)) + 1 bits
                num_bits_consumed = (
                    math.ceil(self.prob_dist.neg_log_probability(decoded_symbol)) + 1
                )
                return decoded_symbol, num_bits_consumed


def test_shannon_fano_elias_coding():
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
        encoder = ShannonFanoEliasEncoder(prob_dist)
        decoder = ShannonFanoEliasDecoder(prob_dist)

        # perform compression
        is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
        assert is_lossless, "Lossless compression failed"
