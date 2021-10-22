from dataclasses import dataclass
from typing import Any
from core.data_stream import BitsDataStream, DataStream
from core.prob_dist import ProbabilityDist
from core.data_compressor import DataCompressor
from core.data_transformer import (
    BitstringToBitsTransformer,
    CascadeTransformer,
    BitsParserTransformer,
    LookupTableTransformer,
)


@dataclass
class HuffmanTree:
    left_child: Any = None
    right_child: Any = None
    id: Any = None
    code: str = ""

    def __hash__(self):
        """
        this allows adding the object to a dictionary as a key
        """
        return hash(self.id)

    @property
    def is_leaf_node(self):
        return (self.left_child is None) and (self.right_child is None)

    @classmethod
    def build_huffman_tree(cls, prob_dist: ProbabilityDist):
        """
        Build the huffman coding tree
        NOTE: Not the most efficient implementation. The insertion/sorting efficiency can be improved

        1. Sort the prob distribution, combine last two symbols into a single symbol
        2. Continue until a single symbol is left
        """
        # create a prob_dist from the HuffmanTree objects
        # NOTE: as the operation of construction of Huffman Tree involves sorting a prob dist,
        # merging nodes etc, they are kind of operations on the probability distribution itself.
        # we overload the "id" field of each Symbol object as a HuffmanTree

        # Lets say we have symbols {1,2,3,4,5,6} with prob {p1, p2,...p6}
        # We first start by initializing a "list" (represented as a ProbabilityDist)
        # {HuffmanTree(id=3): p3, HuffmanTree(id=6): p6, ... }
        prob_dict_tree = {}
        for symbol in prob_dist.symbol_list:
            prob_dict_tree[cls(id=symbol.id)] = symbol.prob

        node_prob_dist = ProbabilityDist(prob_dict_tree)

        while node_prob_dist.size > 1:

            # sort the prob dist (acc to probability values)
            # For example, if we assume the probabilites are p1 < p2 < p3 ... < p6
            # {HuffmanTree(id=6): p6, HuffmanTree(id=5): p5, ... }
            node_prob_dist.sort(reverse=True)

            # get the last two symbols
            last1 = node_prob_dist.pop()
            last2 = node_prob_dist.pop()

            # insert a symbol with the sum of the two probs
            combined_prob = last1.prob + last2.prob
            combined_node = cls(left_child=last1.id, right_child=last2.id)
            node_prob_dist.add(prob=combined_prob, id=combined_node)

        # finally the node_prob_dist should contain a single element
        assert node_prob_dist.size == 1

        # return the huffman tree
        # only one element should remain
        return node_prob_dist.pop().id

    def get_encoding_table(self):
        """
        parse the tree and get the encoding table
        """

        if self.is_leaf_node:
            return {self.id: self.code}

        encoding_table = dict()
        if self.left_child is not None:
            self.left_child.code = self.code + "0"
            left_table_dict = self.left_child.get_encoding_table()
            encoding_table.update(left_table_dict)

        if self.right_child is not None:
            self.right_child.code = self.code + "1"
            right_table_dict = self.right_child.get_encoding_table()
            encoding_table.update(right_table_dict)

        return encoding_table

    def decode_next_symbol(self, data_stream: BitsDataStream, start_ind: int):
        """
        decode function (to be used with BitsParserTransformer)
        """

        # infer the length
        curr_node = self

        # continue decoding until we reach leaf node
        while not curr_node.is_leaf_node:
            bit = data_stream.data_list[start_ind]
            if str(bit) == "0":
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
            start_ind += 1

        # as we reach the leaft node, the decoded symbol is the id of the node
        decoded_symbol = curr_node.id

        # return the decoded symbol and the new index
        return decoded_symbol, start_ind


class HuffmanCoder(DataCompressor):
    """
    Huffman Coder implementation
    """

    def __init__(self, prob_dist: ProbabilityDist):
        # build the huffman tree
        self.huffman_tree = HuffmanTree.build_huffman_tree(prob_dist)
        self.encoder_lookup_table = self.huffman_tree.get_encoding_table()

        # create encoder and decoder transforms
        encoder_transform = CascadeTransformer(
            [
                LookupTableTransformer(self.encoder_lookup_table),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
        decoder_transform = BitsParserTransformer(self.huffman_tree.decode_next_symbol)
        super().__init__(encoder_transform=encoder_transform, decoder_transform=decoder_transform)
