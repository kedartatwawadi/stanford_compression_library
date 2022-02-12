import abc
from dataclasses import dataclass
from core.data_compressor import DataCompressor
from core.data_transformer import (
    BitstringToBitsTransformer,
    CascadeTransformer,
    LookupFuncTransformer,
    BitsParserTransformer,
    LookupTableTransformer,
)
from utils.tree_utils import BinaryNode
from core.data_block import UintDataBlock, BitsDataBlock
from core.util import bitstring_to_uint, uint_to_bitstring
from core.prob_dist import ProbabilityDist


@dataclass
class PrefixFreeTreeNode(BinaryNode):
    code: str = ""  # FIXME: is this field needed?

    def get_encoding_table(self):
        """
        parse the node and get the encoding table
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


class PrefixFreeTree(abc.ABC):
    """
    PrefixFreeTree -> abstract base class to create a PrefixFreeTree.
    """

    def __init__(self, prob_dist: ProbabilityDist):
        """
        create the prefix free tree
        """
        self.prob_dist = prob_dist
        self.root_node = self.build_tree()

    @abc.abstractmethod
    def build_tree(self) -> BinaryNode:
        """
        abstract function -> needs to be implemented by the subclassing class
        """
        raise NotImplementedError

    def print_tree(self):
        self.root_node.print_node()

    def get_encoding_table(self):
        return self.root_node.get_encoding_table()

    def decode_next_symbol(self, data_block: BitsDataBlock, start_ind: int):
        """
        decode function (to be used with BitsParserTransformer)
        """

        # infer the length
        curr_node = self.root_node

        # continue decoding until we reach leaf node
        while not curr_node.is_leaf_node:
            bit = data_block.data_list[start_ind]
            if str(bit) == "0":
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
            start_ind += 1

        # as we reach the leaft node, the decoded symbol is the id of the node
        decoded_symbol = curr_node.id

        # return the decoded symbol and the new index
        return decoded_symbol, start_ind


class PrefixFreeTreeEncoder(DataEncoder):
    pass


class PrefixFreeCoder(DataCompressor):
    """
    Generic Prefix Coder implementation
    """

    def __init__(self, prefix_free_tree: PrefixFreeTree):

        self.prefix_free_tree = prefix_free_tree

        # get the encoding table
        self.encoder_lookup_table = self.prefix_free_tree.get_encoding_table()

        # create encoder and decoder transforms
        encoder_transform = CascadeTransformer(
            [
                LookupTableTransformer(self.encoder_lookup_table),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
        decoder_transform = BitsParserTransformer(self.prefix_free_tree.decode_next_symbol)
        super().__init__(encoder_transform=encoder_transform, decoder_transform=decoder_transform)
