"""Shannon Coding
Shannon coding is a basic prefix free code. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of https://en.wikipedia.org/wiki/Shannon–Fano_coding.

This document uses cumulative probability based method for shannon coding as described in the wiki article above.
"""
from typing import Any, Tuple
from utils.bitarray_utils import float_to_bitarrays, BitArray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeEncoder, PrefixFreeDecoder, PrefixFreeTree, BinaryNode
from core.prob_dist import ProbabilityDist
import math


class ShannonTree(PrefixFreeTree):
    """
    Generates Codewords based on Shannon Coding. Allows generating PrefixFreeTree from the obtained codewords.
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        # sort the probability distribution in decreasing probability and get cumulative probability which will be
        # used for encoding
        self.sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(prob_dist.prob_dict)
        self.cum_prob_dict = self.sorted_prob_dist.cumulative_prob_dict
        # construct the tree and set the root_node of PrefixFreeTree base class
        super().__init__(root_node=self.build_shannon_tree())

    def encode_alphabet(self, symbol) -> BitArray:
        # compute the mid-point corresponding to the range of the given symbol
        cum_prob = self.cum_prob_dict[symbol]

        # compute encode length
        encode_len = math.ceil(self.prob_dist.log_probability(symbol))

        # the encode is the binary representation of the cumulative probability in ascending order,
        _, code = float_to_bitarrays(cum_prob, encode_len)
        return code

    @staticmethod
    def _build_tree_from_code(symbol, code, root_node) -> BinaryNode:
        """ function to generate prefix-free tree from code.
        Args:
            symbol: current symbol
            code: current code
            root_node: root node to the ShannonTree

        Returns:
            the pointer to root node of the tree so far
        """
        # initialize the curr_node, code_so_far temporary var
        curr_node = root_node
        code_so_far = BitArray()
        code_len = len(code)

        for i, bit in enumerate(code):
            # if pointer is None, the object is not updated
            # https://stackoverflow.com/questions/55777748/updating-none-value-does-not-reflect-in-the-object
            if curr_node.right_child is None: curr_node.right_child = BinaryNode(id=None)
            if curr_node.left_child is None: curr_node.left_child = BinaryNode(id=None)

            code_so_far.append(bit)

            # get a pointer to child node
            child = curr_node.right_child if bit else curr_node.left_child

            # if it's the last bit, add the codeword as ID
            if i == (code_len - 1):
                child.id = symbol

            # continue looping through the tree
            curr_node = child
        return root_node

    def build_shannon_tree(self):
        """
        For all symbols in the alphabet, get it's code, and add it to the PrefixFreeTree.
        """
        root_node = BinaryNode(id=None)
        for s in self.sorted_prob_dist.prob_dict:
            code = self.encode_alphabet(s)
            root_node = self._build_tree_from_code(s, code, root_node)
        return root_node


class ShannonEncoder(PrefixFreeEncoder):
    """
    PrefixFreeEncoder already has a encode_block function to encode the symbols once we define a encode_symbol function
    for the particular compressor.
    PrefixFreeTree provides get_encoding_table given a PrefixFreeTree
    """

    def __init__(self, prob_dist: ProbabilityDist):
        tree = ShannonTree(prob_dist)
        self.encoding_table = tree.get_encoding_table()

    def encode_symbol(self, s):
        return self.encoding_table[s]


class ShannonDecoder(PrefixFreeDecoder):
    """
    PrefixFreeDecoder already has a decode_block function to decode the symbols once we define a decode_symbol function
    for the particular compressor.
    PrefixFreeTree provides decode_symbol given a PrefixFreeTree
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = ShannonTree(prob_dist)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


def test_shannon_coding():
    NUM_SAMPLES = 2000
    distributions = [
        ProbabilityDist({"A": 0.5, "B": 0.5}),
        ProbabilityDist({"A": 0.3, "B": 0.3, "C": 0.4}),
        ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.12, "D": 0.13}),
        ProbabilityDist({"A": 0.9, "B": 0.1})
    ]
    expected_codewords = [
        {"A": BitArray('0'), "B": BitArray('1')},
        {"A": BitArray('01'), "B": BitArray('10'), "C": BitArray('00')},
        {"A": BitArray('0'), "B": BitArray('10'), "C": BitArray('1110'), "D": BitArray('110')},
        {"A": BitArray('0'), "B": BitArray('1110')}
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
