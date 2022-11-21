"""File implementing utility abstract classes for prefix free compressors

NOTE: prefix free codes are codes which allow convenient per-symbol encoding/decoding.

We implement PrefixFreeEncoder, PrefixFreeDecoder and PrefixFreeTree which are utility abstract classes
useful for implementing any prefix free code
"""

import abc
from typing import Mapping, Tuple, Any
from utils.bitarray_utils import BitArray
from utils.tree_utils import BinaryNode
from core.data_encoder_decoder import DataEncoder, DataDecoder
from core.data_block import DataBlock


class PrefixFreeEncoder(DataEncoder):
    @abc.abstractmethod
    def encode_symbol(self, s) -> BitArray:
        """
        encode one symbol. method needs to be defined in inherited class.

        Args:
            s (Any): symbol to encode

        Returns:
            BitArray: the encoding for one particular symbol
        """
        pass

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """
        encode the block of data one symbol at a time

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
        """
        decode the next symbol. method needs to be defined in inherited class.

        Args:
            encoded_bitarray (BitArray): _description_

        Returns:
            Tuple[Any, BitArray]: returns the tuple (symbol, bitarray)
        """
        pass

    def decode_block(self, bitarray: BitArray):
        """
        decode the bitarray one symbol at a time using the decode_symbol

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
    """
    Class representing a Prefix Free Tree

    Root node is the pointer to root of the tree with appropriate pointers to the children.
    It subclasses from BinaryNode class in utils/tree_utils which provide a basic binary node with
    left child, right child and id per node pointers.

    Any subclassing class needs to set the root_node appropriately.

    The class also provides method for utilizing the fact that the given tree is PrefixFree and hence we can utilize
    the tree structure to encode and decode. These functions can be used to encode and decode once subclassing function
    for a particular compressor implements the tree generation logic.

    The encode need not require tree structure and hence by default provides a function for getting encoding_table
    given the PrefixFreeTree structure. Decode on the other hand, can always benefit from decoding efficiently using
    the codebook in tree structure.

    In particular,

            get_encoding_table: returns the mapping from tree to the encoding_table for whole codebook which can be
            utilized by PrefixFreeEncoder
            decode_symbol: provides the symbol-by-symbol decoding which can be utilized by the PrefixFreeDecoder
    """

    def __init__(self, root_node: BinaryNode):
        self.root_node = root_node

    def print_tree(self):
        """
        Returns: Visualize tree
        """
        self.root_node.print_node()

    def get_encoding_table(self) -> Mapping[Any, BitArray]:
        """
        Utility func to get the encoding table based on the prefix-free tree.
        Does a DFS over the tree to return the encoding table over the whole symbol dictionary starting from root_node

        Returns:
            Mapping[Any,BitArray]: the encoding_array dict
        """
        encoding_table = {}

        # define the DFS function
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

    def decode_symbol(self, encoded_bitarray):
        """
        Decodes the encoded bitarray stream by decoding symbol by symbol. We parse through the prefix free tree, till
        we reach a leaf node which gives us the decoded symbol ID using prefix-free property of the tree.

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

    @classmethod
    def build_prefix_free_tree_from_code(cls, codes):
        """function to generate prefix-free tree from a dictionary of prefix-free codes
        Args:
            codes: dictionary with symbols as keys and codes as values
            root_node: root node of the prefix free tree
        Returns:
            tree: the PrefixFreeTree
        """

        def _add_tree_nodes_from_code(tree: PrefixFreeTree, symbol: Any, code: BitArray):
            """function to add nodes to a prefix-free tree based on a codeword.
            Args:
                symbol: current symbol
                code: current code
                root_node: root node to the ShannonTree
            """
            # initialize the curr_node
            curr_node = tree.root_node

            # NOTE -> checking if code is a BitArray
            # (if it is a string, the if/else conditioning don't work as expected)
            assert isinstance(code, BitArray), "code should be a bitarray"
            for bit in code:
                if bit:
                    # add a new right node in case it doesn't exist
                    if curr_node.right_child is None:
                        curr_node.right_child = BinaryNode(id=None)
                    curr_node = curr_node.right_child
                else:
                    # add a new right node in case it doesn't exist
                    if curr_node.left_child is None:
                        curr_node.left_child = BinaryNode(id=None)
                    curr_node = curr_node.left_child

            # finally at the end set the id of the leaf node as the symbol
            curr_node.id = symbol

        tree = PrefixFreeTree(BinaryNode())
        for s in codes:
            _add_tree_nodes_from_code(tree, s, codes[s])
        return tree
