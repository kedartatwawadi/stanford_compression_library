"""Shannon Tree Encoder/Decoder
HW1 Q4
"""
from typing import Any, Tuple
from utils.bitarray_utils import BitArray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import (
    PrefixFreeEncoder,
    PrefixFreeDecoder,
    PrefixFreeTree,
)
from core.prob_dist import ProbabilityDist, get_avg_neg_log_prob


class ShannonTreeEncoder(PrefixFreeEncoder):
    """
    PrefixFreeEncoder already has a encode_block function to encode the symbols once we define a encode_symbol function
    for the particular compressor.
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.encoding_table = ShannonTreeEncoder.generate_shannon_tree_codebook(self.prob_dist)

    @classmethod
    def generate_shannon_tree_codebook(cls, prob_dist):
        # sort the probability distribution in decreasing probability
        sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(
            prob_dist.prob_dict, descending=True
        )
        codebook = {}

        ############################################################
        # ADD CODE HERE
        raise NotImplementedError("You need to implement generate_shannon_tree_codebook function in "
                                  "ShannonTreeEncoder class")
        ############################################################

        return codebook

    def encode_symbol(self, s):
        return self.encoding_table[s]


class ShannonTreeDecoder(PrefixFreeDecoder):

    def __init__(self, prob_dist: ProbabilityDist):
        encoding_table = ShannonTreeEncoder.generate_shannon_tree_codebook(prob_dist)
        self.tree = PrefixFreeTree.build_prefix_free_tree_from_code(encoding_table)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


def test_shannon_tree_coding_specific_case():
    # NOTE -> this test must succeed with your implementation
    ############################################################
    # Add the computed expected codewords for distributions presented in part 1 to these list to improve the test
    raise NotImplementedError("Add the computed expected codewords for distributions presented in part 1 "
                              "to these list to improve the test")
    ############################################################

    def test_encoded_symbol(prob_dist, expected_codeword_dict):
        """
        test if the encoded symbol is as expected
        """
        encoder = ShannonTreeEncoder(prob_dist)
        for s in prob_dist.prob_dict.keys():
            assert encoder.encode_symbol(s) == expected_codeword_dict[s]

    for i, prob_dist in enumerate(distributions):
        test_encoded_symbol(prob_dist, expected_codeword_dict=expected_codewords[i])


def test_shannon_tree_coding_end_to_end():
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
    ]

    def test_end_to_end(prob_dist, num_samples):
        """
        Test if decoding of (encoded symbol) results in original
        """
        # generate random data
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)

        # create encoder decoder
        encoder = ShannonTreeEncoder(prob_dist)
        decoder = ShannonTreeDecoder(prob_dist)

        # perform compression
        is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder)
        assert is_lossless, "Lossless compression failed"

        # avg_log_prob should be close to the avg_codelen
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        avg_codelen = encode_len / data_block.size
        assert avg_codelen <= (avg_log_prob + 1), "avg_codelen should be within 1 bit of mean_neg_log_prob"
        print(f"Shannon-tree coding: avg_log_prob={avg_log_prob:.3f}, avg codelen: {avg_codelen:.3f}")

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)