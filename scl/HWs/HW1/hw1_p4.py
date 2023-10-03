"""Shannon Tree Encoder/Decoder
HW1 Q4
"""
from typing import Any, Tuple
from scl.utils.bitarray_utils import BitArray
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.compressors.prefix_free_compressors import (
    PrefixFreeEncoder,
    PrefixFreeDecoder,
    PrefixFreeTree,
)
from scl.core.prob_dist import ProbabilityDist, get_avg_neg_log_prob
import math
from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint


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
        """
        :param prob_dist: ProbabilityDist object
        :return: codebook: dictionary mapping symbols to bitarrays
        """

        # sort the probability distribution in decreasing probability
        sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(
            prob_dist.prob_dict, descending=True
        )
        codebook = {}

        ############################################################
        # ADD CODE HERE
        # NOTE:
        # - The utility functions encoding_table.values(), bitarray_to_uint, uint_to_bitarray might be useful

        raise NotImplementedError

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


class ShannonTableDecoder(PrefixFreeDecoder):

    def __init__(self, prob_dist: ProbabilityDist):
        self.decoding_table, self.codelen_table, self.max_codelen = self.create_decoding_table(prob_dist)

    @staticmethod
    def create_decoding_table(prob_dist: ProbabilityDist):
        """
        :param prob_dist: ProbabilityDist object
        :return:
            decoding_table: dictionary mapping bitarrays to symbols
            codelen_table: dictionary mapping symbols to code-length
            max_codelen: maximum code-length of any symbol in the codebook
        """
        # create the encoding table
        encoding_table = ShannonTreeEncoder.generate_shannon_tree_codebook(prob_dist)
        ############################################################
        # ADD CODE HERE
        # NOTE:
        # - The utility functions ProbabilityDist.neg_log_probability
        # - scl.utils.bitarray_utils.uint_to_bitarray and scl.utils.bitarray_utils.bitarry_to_uint might be useful
        raise NotImplementedError
        ############################################################

        return decoding_table, codelen_table, max_codelen

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        # get the padded codeword to be decoded
        padded_codeword = encoded_bitarray[:self.max_codelen]
        if len(encoded_bitarray) < self.max_codelen:
            padded_codeword = padded_codeword + "0" * (self.max_codelen - len(encoded_bitarray))

        decoded_symbol = self.decoding_table[str(padded_codeword)]
        num_bits_consumed = self.codelen_table[decoded_symbol]
        return decoded_symbol, num_bits_consumed


def test_shannon_tree_coding_specific_case():
    # NOTE -> this test must succeed with your implementation
    ############################################################
    # Add the computed expected codewords for distributions presented in part 1 to these list to improve the test
    raise NotImplementedError
    ############################################################

    def test_encoded_symbol(prob_dist, expected_codeword_dict):
        """
        test if the encoded symbol is as expected
        :type prob_dist: ProbabilityDist object
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
        decoder_tree = ShannonTreeDecoder(prob_dist)
        decoder_table = ShannonTableDecoder(prob_dist)

        # perform compression
        is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder_tree)
        assert is_lossless, "Lossless compression failed"

        # avg_log_prob should be close to the avg_codelen
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        avg_codelen = encode_len / data_block.size
        assert avg_codelen <= (avg_log_prob + 1), "avg_codelen should be within 1 bit of mean_neg_log_prob"
        print(f"Shannon-tree coding: avg_log_prob={avg_log_prob:.3f}, avg codelen: {avg_codelen:.3f}")

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)


def test_shannon_table_coding_end_to_end():
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
        decoder_table = ShannonTableDecoder(prob_dist)

        # perform compression
        is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder_table)
        assert is_lossless, "Lossless compression failed"

        # avg_log_prob should be close to the avg_codelen
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        avg_codelen = encode_len / data_block.size
        assert avg_codelen <= (avg_log_prob + 1), "avg_codelen should be within 1 bit of mean_neg_log_prob"
        print(f"Shannon-tree coding: avg_log_prob={avg_log_prob:.3f}, avg codelen: {avg_codelen:.3f}")

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)
