from dataclasses import dataclass
from typing import Any
import heapq
from functools import total_ordering
from compressors.prefix_free_compressors import PrefixFreeCoder, PrefixFreeNode, PrefixFreeTree
from core.prob_dist import ProbabilityDist


@dataclass
@total_ordering  # decorator which adds other compare ops give one
class HuffmanNode(PrefixFreeNode):
    """
    NOTE: PrefixFreeNode class already has left_child, right_child, id, code fields
    here by subclassing we add a couple of more fields: prob
    """

    prob: float = None

    def __le__(self, other):
        """
        Define a comparison operator, so that we can use this while comparing nodes
        # NOTE: we only need to define one compare op, as others can be implemented using the
        decorator @total_ordering
        """
        return self.prob <= other.prob


class HuffmanTree(PrefixFreeTree):
    def build_tree(self):
        # builds the huffman tree and returns the root_node
        return self.build_huffman_tree(self.prob_dist)

    @staticmethod
    def build_huffman_tree(prob_dist: ProbabilityDist) -> HuffmanNode:
        """
        Build the huffman coding tree

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

            # get the two smallest symbols
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


class HuffmanCoder(PrefixFreeCoder):
    """
    Huffman Coder implementation
    """

    def __init__(self, prob_dist: ProbabilityDist):
        # build the huffman tree
        self.huffman_tree = HuffmanTree(prob_dist)

        # initialize the params of the super class (PrefixFreeCoder)
        super().__init__(self.huffman_tree)
