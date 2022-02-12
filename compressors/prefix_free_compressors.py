from dataclasses import dataclass
from typing import Mapping
from utils.bitarray_utils import BitArray
from utils.tree_utils import BinaryNode
from core.prob_dist import ProbabilityDist
from core.data_encoder_decoder import DataEncoder, DataDecoder
from core.data_block import DataBlock


@dataclass
class PrefixFreeTreeNode(BinaryNode):
    code: str = BitArray("")  # FIXME: is this field needed?

    def get_encoding_table(self) -> Mapping[str, BitArray]:
        """
        parse the node and get the encoding table
        """

        if self.is_leaf_node:
            return {self.id: self.code}

        encoding_table = dict()
        if self.left_child is not None:
            self.left_child.code = self.code + BitArray("0")
            left_table_dict = self.left_child.get_encoding_table()
            encoding_table.update(left_table_dict)

        if self.right_child is not None:
            self.right_child.code = self.code + BitArray("1")
            right_table_dict = self.right_child.get_encoding_table()
            encoding_table.update(right_table_dict)

        return encoding_table


class PrefixFreeTree:
    def __init__(self, prob_dist: ProbabilityDist):
        """
        create the prefix free tree
        """
        self.root_node = self.build_tree(prob_dist)

    def get_encoding_table(self):
        return self.root_node.get_encoding_table()

    def print_tree(self):
        self.root_node.print_node()

    @staticmethod
    def build_tree(prob_dist) -> PrefixFreeTreeNode:
        """
        abstract function -> needs to be implemented by the subclassing class
        """
        raise NotImplementedError


class PrefixFreeTreeEncoder(DataEncoder):
    def __init__(self, prob_dist: ProbabilityDist):
        """
        create the prefix free tree
        """
        self.tree = PrefixFreeTree(prob_dist)

    def encode_symbol(self, s):
        # initialize the encoding table once, if it has not been created
        if not hasattr(self, "encoding_table"):
            self.encoding_table = self.tree.get_encoding_table()

        return self.encoding_table[s]

    def encode_block(self, data_block: DataBlock):
        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class PrefixFreeTreeDecoder(DataDecoder):
    def __init__(self, prob_dist: ProbabilityDist):
        """
        create the prefix free tree
        """
        self.tree = PrefixFreeTree(prob_dist)

    def decode_symbol(self, encoded_bitarray):
        # initialize num_bits_consumed
        num_bits_consumed = 0

        # continue decoding until we reach leaf node
        curr_node = self.tree.root_node
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

    def decode_block(self, bitarray: BitArray):
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed
