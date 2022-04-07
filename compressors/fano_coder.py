"""Shannon Coding
Shannon coding is a basic prefix free coding. There is some confusion in literature about Shannon, Fano and
Shannon-Fano codes, e.g. see Naming section of https://en.wikipedia.org/wiki/Shannon–Fano_coding.

This document uses Fano coding as described in the wiki article above.
"""
from utils.bitarray_utils import float_to_bitarrays, BitArray, uint_to_bitarray
from utils.test_utils import get_random_data_block, try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeTree, PrefixFreeTreeNode, PrefixFreeTreeEncoder, PrefixFreeTreeDecoder
from core.prob_dist import ProbabilityDist
import math


class FanoTree(PrefixFreeTree):
    def __init__(self, prob_dist):
        super().__init__(prob_dist)
        self.sorted_prob_dist = prob_dist.sorted_prob_list
        self.root_node = PrefixFreeTreeNode(id=None)

        # build tree returns the root node
        self.build_tree(self.root_node, self.sorted_prob_dist)

    @staticmethod
    def normalize_prob_dict(prob_dict):
        sum_p = sum(prob_dict.values())
        return ProbabilityDist(dict([[a, b/sum_p] for a, b in prob_dict.items()]))

    @staticmethod
    def build_tree(root_node, norm_sort_prob_list) -> PrefixFreeTreeNode:
        cumulative_prob_dict = norm_sort_prob_list.cumulative_prob_dict

        left_prob_dict = {}
        right_prob_dict = {}

        if norm_sort_prob_list.size == 2:
            left_symbol, right_symbol = norm_sort_prob_list.alphabet
            root_node.left_child = PrefixFreeTreeNode(id=left_symbol)
            root_node.right_child = PrefixFreeTreeNode(id=right_symbol)
            return root_node

        for s, cum_prob in cumulative_prob_dict.items():
            if cum_prob < 0.5:
                left_prob_dict.update({s: norm_sort_prob_list.probability(s)})
            else:
                right_prob_dict.update({s: norm_sort_prob_list.probability(s)})

        if len(left_prob_dict) != 1:
            root_node.left_child = PrefixFreeTreeNode(id=None)
            FanoTree.build_tree(root_node.left_child, FanoTree.normalize_prob_dict(left_prob_dict))
        else:
            root_node.left_child = PrefixFreeTreeNode(id=list(left_prob_dict)[0])

        if len(right_prob_dict) != 1:
            root_node.right_child = PrefixFreeTreeNode(id=None)
            FanoTree.build_tree(root_node.right_child, FanoTree.normalize_prob_dict(right_prob_dict))
        else:
            root_node.right_child = PrefixFreeTreeNode(id=list(right_prob_dict)[0])

        return root_node


class FanoEncoder(PrefixFreeTreeEncoder):
    """
    Outputs codewords
    """

    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = FanoTree(prob_dist)


class FanoDecoder(PrefixFreeTreeDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        self.tree = FanoTree(prob_dist)


def test_fano_coding():
    """test if shannon fano encoding is lossless"""
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
        encoder = FanoEncoder(prob_dist)
        decoder = FanoDecoder(prob_dist)

        # perform compression
        is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
        assert is_lossless, "Lossless compression failed"
