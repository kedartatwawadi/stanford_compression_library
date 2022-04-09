"""File implementing utility abstract classes for prefix free compressors

NOTE: prefix free codes are codes which allow convenient per-symbol encoding/decoding.

We implement PrefixFreeEncoder, PrefixFreeDecoder
and PrefixFreeTreeEncoder, PrefixFreeTreeDecoder which are utility abstract classes
useful for implementing any prefix free code
"""

import abc
from dataclasses import dataclass
from typing import Mapping, Tuple, Any
from utils.bitarray_utils import BitArray
from utils.tree_utils import BinaryNode
from core.prob_dist import ProbabilityDist
from core.data_encoder_decoder import DataEncoder, DataDecoder
from core.data_block import DataBlock


class PrefixFreeEncoder(DataEncoder):
    @abc.abstractmethod
    def encode_symbol(self, s) -> BitArray:
        """encode one symbol

        Args:
            s (Any): symbol to encode

        Returns:
            BitArray: the encoding for one particular symbol
        """

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """encode the block of data one symbol at a time

        prefix free codes have specific code for each symbol, we implement encode_block
        function as a simple loop over encode_symbol function.
        This class can also be used by certain non-prefix free codes which support symbol-wise encoding

        Args:
            data_block (DataBlock): input block to encoded
        Returns:
            BitArray: encoded bitarray
        """

        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class PrefixFreeDecoder(DataDecoder):
    @abc.abstractmethod
    def decode_symbol(self, encoded_bitarray: BitArray) -> Tuple[Any, BitArray]:
        """decode the next symbol

        Args:
            encoded_bitarray (BitArray): _description_

        Returns:
            Tuple[Any, BitArray]: returns the tuple (symbol, bitarray)
        """
        pass

    def decode_block(self, bitarray: BitArray):
        """decode the bitarray one symbol at a time using the decode_symbol

        as prefix free codes have specific code for each symbol, and due to the prefix free nature, allow for
        decoding each symbol from the stream, we implement decode_block function as a simple loop over
        decode_symbol function.

        Args:
            bitarray (BitArray): input bitarray with encoding of >=1 integers

        Returns:
            Tuple[DataBlock, Int]: return decoded integers in data block, number of bits read from input
        """
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed


class PrefixFreeTree:
    def __init__(self, prob_dist: ProbabilityDist):
        """
        create the prefix free tree
        """
        self.prob_dist = prob_dist
        self.root_node = self.build_tree()

    def get_encoding_table(self):
        """
        parse the root node and get the encoding table
        """
        encoding_table = {}

        # define the BFS function
        def _parse_node(node: BinaryNode, code: BitArray):
            """parse the node in DFS fashion, and get the code corresponding to
            all the leaf nodes

            Args:
                node (BinaryNode): the current node being parsed
                code (BitArray): the code corresponding to the current node
            """
            # if node is leaf add it to the table
            if node.is_leaf_node:
                encoding_table[node.id] = code

            if node.left_child is not None:
                _parse_node(node.left_child, code + BitArray("0"))

            if node.right_child is not None:
                _parse_node(node.right_child, code + BitArray("1"))

        # call the parsing function on the root node
        _parse_node(self.root_node, BitArray(""))
        return encoding_table

    def print_tree(self):
        self.root_node.print_node()

    def build_tree(self) -> BinaryNode:
        """
        abstract function -> needs to be implemented by the subclassing class
        """
        raise NotImplementedError

    def decode_symbol(self, encoded_bitarray):
        """decode each symbol by parsing through the prefix free tree

        - start from the root node
        - if the next bit is 0, go left, else right
        - once you reach a leaf node, output the symbol corresponding the node
        """
        # initialize num_bits_consumed
        num_bits_consumed = 0

        # continue decoding until we reach leaf node
        curr_node = self.root_node
        while not curr_node.is_leaf_node:
            bit = encoded_bitarray[num_bits_consumed]
            if bit == 0:
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
            num_bits_consumed += 1

        # as we reach the leaf node, the decoded symbol is the id of the node
        decoded_symbol = curr_node.id
        return decoded_symbol, num_bits_consumed
