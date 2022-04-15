"""Shannon Coding
Shannon coding is a basic prefix free code. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of https://en.wikipedia.org/wiki/Shannon–Fano_coding.

This document uses cumulative probability based method for shannon coding as described in the wiki article above.
"""
from dataclasses import dataclass
from typing import Any, Tuple
from utils.bitarray_utils import float_to_bitarrays, BitArray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeEncoder, PrefixFreeDecoder, PrefixFreeTree, BinaryNode
from core.prob_dist import ProbabilityDist
import math


@dataclass
class ShannonNode(BinaryNode):
    code: BitArray = None


class ShannonTree(PrefixFreeTree):
    def __init__(self, prob_dist: ProbabilityDist):
        self.prob_dist = prob_dist
        self.sorted_prob_dist = ProbabilityDist.get_sorted_prob_dist(prob_dist.prob_dict)
        self.cum_prob_dict = self.sorted_prob_dist.cumulative_prob_dict
        # construct the tree and set the root_node of PrefixFreeTree base class
        super().__init__(root_node=self.build_shannon_tree())

    def encode_symbol(self, symbol) -> BitArray:
        # compute the mid-point corresponding to the range of the given symbol
        cum_prob = self.cum_prob_dict[symbol]

        # compute encode length
        encode_len = math.ceil(self.prob_dist.log_probability(symbol))

        # the encode is the binary representation of the cumulative probability in ascending order,
        _, code = float_to_bitarrays(cum_prob, encode_len)
        return code

    @staticmethod
    def build_tree_from_code(symbol, code, root_node) -> BinaryNode:
        code_so_far = BitArray()
        code_len = len(code)

        for i, bit in enumerate(code):
            code_so_far.append(bit)

            if i == 0:
                curr_node = root_node
                right_child = curr_node.right_child
                left_child = curr_node.left_child

            if i == (code_len - 1):
                next_node = ShannonNode(id=symbol, code=code_so_far)
            else:
                next_node = ShannonNode(id=None, code=code_so_far)

            if bit:
                if right_child is None:
                    curr_node.right_child = next_node
                else:
                    next_node = right_child
                    curr_node.right_child = next_node
            else:
                if left_child is None:
                    curr_node.left_child = next_node
                else:
                    next_node = left_child
                    curr_node.left_child = next_node

            curr_node = next_node
            right_child = curr_node.right_child
            left_child = curr_node.left_child

        return root_node

    def build_shannon_tree(self):
        root_node = ShannonNode(id=None, code=None)
        for s in self.sorted_prob_dist.prob_dict:
            code = self.encode_symbol(s)
            root_node = self.build_tree_from_code(s, code, root_node)

        return root_node


class ShannonEncoder(PrefixFreeEncoder):
    """
    Outputs codewords
    """

    def __init__(self, prob_dist: ProbabilityDist):
        tree = ShannonTree(prob_dist)
        self.encoding_table = tree.get_encoding_table()

    def encode_symbol(self, s):
        return self.encoding_table[s]


class ShannonDecoder(PrefixFreeDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = ShannonTree(prob_dist)

    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        decoded_symbol, num_bits_consumed = self.tree.decode_symbol(encoded_bitarray)
        return decoded_symbol, num_bits_consumed


def test_shannon_coding():
    """test if shannon encoding is lossless"""
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
