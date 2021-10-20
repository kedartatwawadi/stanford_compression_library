from dataclasses import dataclass
from typing import Any
from core.prob_dist import ProbabilityDist
from core.data_compressor import DataCompressor
from core.data_transformer import (
    BitstringToBitsTransformer,
    CascadeTransformer,
    LookupFuncTransformer,
)
from core.misc_transformers import BitsParserTransformer


@dataclass
class HuffmanTree:
    left_child: Any = None
    right_child: Any = None
    id: Any = None

    def get_encoding_table(self):
        """
        perform DFS graph traversal to get the code (any traversal will do)
        """
        pass


def build_huffman_tree(prob_dist: ProbabilityDist):
    """
    Build the huffman coding tree
    NOTE: Not the most efficient implementation. The insertion/sorting efficiency can be improved 

    1. Sort the prob distribution, combine last two symbols into a single symbol
    2. Continue until a single symbol is left
    """
    # create a prob_dist from the HuffmanTree objects
    prob_dict_tree = {}
    for symbol in prob_dist.symbol_list:
        prob_dict_tree[HuffmanTree(id=symbol.id)] = symbol.prob

    node_prob_dist = ProbabilityDist(prob_dict_tree)

    while node_prob_dist.size > 1:

        # sort the prob dist
        node_prob_dist.sort()

        # get the last two symbols
        last1 = node_prob_dist.pop()
        last2 = node_prob_dist.pop()

        # insert a symbol with the sum of the two probs
        combined_prob = last1.prob+last2.prob
        combined_node = HuffmanTree(left_child=last1, right_child=last2)
        node_prob_dist.add(prob=combined_prob, id=combined_node)


    # finally the node_prob_dist should contain a single element
    assert len(node_prob_dist) == 1
    ProbabilityDist._validate_prob_dist(node_prob_dist)

    # return the huffman tree
    return node_prob_dist.pop()




class HuffmanCoder(DataCompressor):
    """
    TODO:
    """

    def __init__(self, prob_dist: ProbabilityDist):

        # build the huffman tree
        self.huffman_tree = self.build_huffman_tree(prob_dist)

        #

    @staticmethod
    def build_huffman_tree(prob_dist: ProbabilityDist):
        """
        Build the huffman coding tree
        NOTE: Not the most efficient implementation. The insertion/sorting efficiency can be improved 

        1. Sort the prob distribution, combine last two symbols into a single symbol
        2. Continue until a single symbol is left
        """
        # create a prob_dist from the HuffmanTree objects
        prob_dict_tree = {}
        for symbol in prob_dist.symbol_list:
            prob_dict_tree[HuffmanTree(id=symbol.id)] = symbol.prob

        node_prob_dist = ProbabilityDist(prob_dict_tree)

        while node_prob_dist.size > 1:

            # sort the prob dist
            node_prob_dist.sort()

            # get the last two symbols
            last1 = node_prob_dist.pop()
            last2 = node_prob_dist.pop()

            # insert a symbol with the sum of the two probs
            combined_prob = last1.prob+last2.prob
            combined_node = HuffmanTree(left_child=last1, right_child=last2)
            node_prob_dist.add(prob=combined_prob, id=combined_node)


        # finally the node_prob_dist should contain a single element
        assert len(node_prob_dist) == 1
        ProbabilityDist._validate_prob_dist(node_prob_dist)

        # return the huffman tree
        return node_prob_dist.pop()
    


    @staticmethod
    def encoder_lookup_func(x: int):
        assert x >= 0
        assert isinstance(x, int)

        bitstring = uint_to_bitstring(x)
        len_bitstring = len(bitstring) * "1" + "0"

        return len_bitstring + bitstring

    @staticmethod
    def decoder_bits_parser(data_stream, start_ind):

        # infer the length
        num_ones = 0
        for ind in range(start_ind, data_stream.size):
            bit = data_stream.data_list[ind]
            if str(bit) == "0":
                break
            num_ones += 1

        # compute the new start_ind
        new_start_ind = 2 * num_ones + 1 + start_ind

        # decode the symbol
        bitstring = "".join(data_stream.data_list[start_ind + num_ones + 1 : new_start_ind])
        symbol = bitstring_to_uint(bitstring)

        return symbol, new_start_ind

    def set_encoder_decoder_params(self, data_stream):

        # create encoder and decoder transforms
        self.encoder_transform = CascadeTransformer(
            [
                LookupFuncTransformer(self.encoder_lookup_func),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
        self.decoder_transform = BitsParserTransformer(self.decoder_bits_parser)

