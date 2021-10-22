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
    prob: float = None

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
        # Lets say we have symbols {1,2,3,4,5,6} with prob {p1, p2,...p6}
        # We first start by initializing a list
        # [ HuffmanTree(id=3, prob=p3), (HuffmanTree(id=6, prob=p6),  ]

        node_list = []
        for a in prob_dist.alphabet:
            node = cls(id=a, prob=prob_dist.probability(a))
            node_list.append(node)

        while len(node_list) > 1:

            # sort the prob dist in the reverse order(acc to probability values)
            # For example, if we assume the probabilites are p1 < p2 < p3 ... < p6
            # [HuffmanTree(id=6, prob=p6), HuffmanTree(id=5, prob=p5), ... ]
            node_list.sort(key=lambda x: -x.prob)

            # get the last two symbols
            last1 = node_list.pop()
            last2 = node_list.pop()

            # insert a symbol with the sum of the two probs
            combined_prob = last1.prob + last2.prob
            combined_node = cls(left_child=last1, right_child=last2, prob=combined_prob)
            node_list.append(combined_node)

        # finally the node_prob_dist should contain a single element
        assert len(node_list) == 1

        # return the huffman tree
        # only one element should remain
        huffman_tree = node_list[0]
        return huffman_tree

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
