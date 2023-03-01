"""Shannon Coding
Shannon coding is a basic prefix free code. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of
https://en.wikipedia.org/w/index.php?title=Shannon–Fano_coding&oldid=1076520027.

This document uses cumulative probability based method for shannon coding as described in the wiki article above.

The Shannon code construction is as follows:

a. Given the probabilities $p_1, p_2, \ldots, p_k$, sort them in the descending order. WLOG let $$ p_1 \geq p_2 \geq \ldots \geq p_k$$
b. compute the cumulative probabilities $c_1, c_2, \ldots, c_k$ such that:
   $$ \begin{aligned}
   c_1 &= 0 \\
   c_2 &= p_1 \\
   c_3 &= p_1 + p_2 \\
   &\ldots \\
   c_k &= p_1 + p_2 + \ldots + p_{k-1}
   \end{aligned} $$
c. Note that we can represent any real number between $[0,1)$ in binary as $b0.b_1 b_2 b_3 \ldots $, where $b_1, b_2,
   b_3, \ldots$ are some bits. For example:
   $$ \begin{aligned}
   0.5 &= b0.1 \\
   0.25 &= b0.01 \\
   0.3 &= b0.010101...
   \end{aligned} $$ This is very similar to how we represent real numbers using "decimal" floating point value, but it
   is using "binary" floating point values (This is actually similar to how computers represent floating point numbers
   internally!)
d. If the "binary" floating point representation is clear, then the Shannon code for symbol $r$ of codelength $ l_r =
   \left\lceil \log_2 \frac{1}{p_r} \right\rceil $ can be obtained by simply truncating the binary floating point
   representation of $c_r$
"""
from typing import Any, Tuple
from scl.utils.bitarray_utils import float_to_bitarrays, BitArray
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.compressors.prefix_free_compressors import (
    PrefixFreeEncoder,
    PrefixFreeDecoder,
    PrefixFreeTree,
)
from scl.core.prob_dist import ProbabilityDist
import math


class ShannonEncoder(PrefixFreeEncoder):
    """
    PrefixFreeEncoder already has a encode_block function to encode the symbols once we define a encode_symbol function
    for the particular compressor.
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.encoding_table = ShannonEncoder.generate_shannon_codebook(self.prob_dist)

    @classmethod
    def generate_shannon_codebook(cls, prob_dist):
        # sort the probability distribution in decreasing probability and get cumulative probability which will be
        # used for encoding
        sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(
            prob_dist.prob_dict, descending=True
        )
        cum_prob_dict = sorted_prob_dist.cumulative_prob_dict

        codebook = {}
        for s in sorted_prob_dist.prob_dict:
            # get the encode length for the symbol s
            encode_len = math.ceil(sorted_prob_dist.neg_log_probability(s))

            # get the code as a truncated floating point representation
            _, code = float_to_bitarrays(cum_prob_dict[s], encode_len)
            codebook[s] = code
        return codebook

    def encode_symbol(self, s):
        return self.encoding_table[s]


class ShannonDecoder(PrefixFreeDecoder):
    """
    PrefixFreeDecoder already has a decode_block function to decode the symbols once we define a decode_symbol function
    for the particular compressor.
    PrefixFreeTree provides decode_symbol given a PrefixFreeTree
    """

    def __init__(self, prob_dist: ProbabilityDist):
        encoding_table = ShannonEncoder.generate_shannon_codebook(prob_dist)
        self.tree = PrefixFreeTree.build_prefix_free_tree_from_code(encoding_table)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


def test_shannon_coding():
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.3, "B": 0.3, "C": 0.4}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1}),
    ]
    expected_codewords = [
        {"A": BitArray("0"), "B": BitArray("1")},
        {"A": BitArray("01"), "B": BitArray("10"), "C": BitArray("00")},
        {"A": BitArray("0"), "B": BitArray("10"), "C": BitArray("1110"), "D": BitArray("110")},
        {"A": BitArray("0"), "B": BitArray("1110")},
    ]

    def test_end_to_end(prob_dist, num_samples):
        """
        Test if decoding of (encoded symbol) results in original
        """
        # generate random data
        data_block = get_random_data_block(prob_dist, num_samples, seed=0)

        # create encoder decoder
        encoder = ShannonEncoder(prob_dist)
        decoder = ShannonDecoder(prob_dist)

        # perform compression
        is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
        assert is_lossless, "Lossless compression failed"

    def test_encoded_symbol(prob_dist, expected_codeword_dict):
        """
        test if the encoded symbol is as expected
        """
        encoder = ShannonEncoder(prob_dist)
        for s in prob_dist.prob_dict.keys():
            assert encoder.encode_symbol(s) == expected_codeword_dict[s]

    for i, prob_dist in enumerate(distributions):
        test_end_to_end(prob_dist, NUM_SAMPLES)
        test_encoded_symbol(prob_dist, expected_codeword_dict=expected_codewords[i])
