from dataclasses import dataclass
from typing import Any
import heapq
from functools import total_ordering
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
@total_ordering  # decorator which adds other compare ops give one
class HuffmanNode:
    left_child: Any = None
    right_child: Any = None
    id: Any = None
    code: str = ""
    prob: float = None

    @property
    def is_leaf_node(self):
        return (self.left_child is None) and (self.right_child is None)

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

    def __le__(self, other):
        """
        Define a comparison operator, so that we can use this while comparing nodes
        # NOTE: we only need to define one compare op, as others can be implemented using the
        decorator @total_ordering
        """
        return self.prob <= other.prob


class HuffmanTree:
    def __init__(self, prob_dist: ProbabilityDist):
        """
        create the huffman tree
        """
        self.prob_dist = prob_dist
        self.root_node = self.build_huffman_tree(self.prob_dist)

    @staticmethod
    def build_huffman_tree(prob_dist: ProbabilityDist) -> HuffmanNode:
        """
        Build the huffman coding tree
        NOTE: Not the most efficient implementation. The insertion/sorting efficiency can be improved

        1. Sort the prob distribution, combine last two symbols into a single symbol
        2. Continue until a single symbol is left
        """
        # Lets say we have symbols {1,2,3,4,5,6} with prob {p1, p2,...p6}
        # We first start by initializing a list
        # [ HuffmanNode(id=3, prob=p3), (HuffmanTree(id=6, prob=p6),  ]

        node_list = []
        for a in prob_dist.alphabet:
            node = HuffmanNode(id=a, prob=prob_dist.probability(a))
            node_list.append(node)

        # create a node_heap from the node_list (in place)
        # NOTE: We create a min-heap data structure to represent the list, as
        # We are concerned about finding the top two smallest elements from the list
        # Heaps are efficient at such operations O(log(n)) -> push/pop, O(1) -> min val
        node_heap = node_list  # shallow copy
        heapq.heapify(node_heap)

        while len(node_heap) > 1:
            # create a min-heap (in-place) from the node list, so that we can get the
            heapq.heapify(node_heap)

            # get the last two symbols
            last1 = heapq.heappop(node_heap)
            last2 = heapq.heappop(node_heap)

            # insert a symbol with the sum of the two probs
            combined_prob = last1.prob + last2.prob
            combined_node = HuffmanNode(left_child=last1, right_child=last2, prob=combined_prob)
            heapq.heappush(node_heap, combined_node)

        # finally the node_prob_dist should contain a single element
        assert len(node_heap) == 1

        # return the huffman tree
        # only one element should remain
        root_node = node_heap[0]
        return root_node

    def get_encoding_table(self):
        return self.root_node.get_encoding_table()

    def decode_next_symbol(self, data_stream: BitsDataStream, start_ind: int):
        """
        decode function (to be used with BitsParserTransformer)
        """

        # infer the length
        curr_node = self.root_node

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
        self.huffman_tree = HuffmanTree(prob_dist)
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
