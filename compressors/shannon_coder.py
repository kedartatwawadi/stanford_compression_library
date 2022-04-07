"""Shannon Coding
Shannon coding is a basic prefix free coding. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of https://en.wikipedia.org/wiki/Shannon–Fano_coding.

This document uses cumulative probability based method for shannon coding as described in the wiki article above.
"""
from utils.bitarray_utils import float_to_bitarrays, BitArray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeEncoder, PrefixFreeTree, PrefixFreeTreeDecoder
from core.prob_dist import ProbabilityDist
import math


class ShannonEncoder(PrefixFreeEncoder):
    """
    Outputs codewords
    """
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

        print(f"Encoder: symbol = {symbol}, code = {code}")
        return code


class ShannonDecoder(PrefixFreeTreeDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.sorted_prob_dist = prob_dist.sorted_prob_list
        self.cumulative_prob_dict = self.sorted_prob_dist.cumulative_prob_dict
        self.tree = PrefixFreeTree(prob_dist)

        self.create_shannon_tree(prob_dist)

    def get_codeword(self, symbol) -> BitArray:
        # compute the mid-point corresponding to the range of the given symbol
        cum_prob = self.cumulative_prob_dict[symbol]
        # compute encode length
        encode_len = math.ceil(self.prob_dist.log_probability(symbol))
        # the encoded value is the binary representation of the cumulative probability in ascending order,
        # truncated to expected code-word length
        _, code = float_to_bitarrays(cum_prob, encode_len)
        return code

    def create_shannon_tree(self, prob_dist):
        encoding_table = {}
        for s in prob_dist.prob_dict.keys():
            encoding_table.update({s: self.get_codeword(s)})

        print(f"Decoder: Encoding Table = {encoding_table}")

        self.tree.build_tree_from_encoding_table(encoding_table)


def test_shannon_coding():
    """test if shannon fano elais encoding is lossless"""
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.3, "B": 0.3, "C": 0.4}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
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
