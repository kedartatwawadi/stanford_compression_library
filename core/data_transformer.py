"""
DataTransformer -> takes in an input DataBlock as input and outputs another DataBlock.
DataTransformers can be thought of as "stages" or "layers" which we can appropriately combine
to form the data compression encoder/decoders.
"""

import abc
from core.data_block import (
    BitsDataBlock,
    BitstringDataBlock,
    DataBlock,
    StringDataBlock,
    UintDataBlock,
)
from typing import Callable, List, Dict

from core.util import bitstring_to_uint, uint_to_bitstring


class DataTransformer(abc.ABC):
    """
    a DataTransformer transform the input DataBlock into an output DataBlock
    """

    @abc.abstractmethod
    def transform(self, data_block: DataBlock) -> DataBlock:
        """
        NOTE: For any DataTransformer, the transform function needs to be implemented
        the transform function of every DataTransformer takes only the data_block as input
        """
        return None


class IdentityTransformer(DataTransformer):
    """
    returns the input data block as is
    """

    def transform(self, data_block: DataBlock) -> DataBlock:
        return data_block


class SplitStringTransformer(DataTransformer):
    """
    assumes input symbol is a string, and splits the string into individual chars
    For eg:
    ["aa", "aab"] -> ["a","a","a","a","b]
    """

    def transform(self, data_block: StringDataBlock) -> DataBlock:
        output_list = []
        for symbol in data_block.data_list:
            # check if the symbol is valid
            assert self.is_symbol_valid(symbol)

            output_list += [c for c in symbol]
        return StringDataBlock(output_list)

    @staticmethod
    def is_symbol_valid(symbol) -> bool:
        return StringDataBlock.validate_data_symbol(symbol)


class BitstringToBitsTransformer(SplitStringTransformer):
    """
    splits the input bitstring List into a list of bits
    eg: ["00", "001"] -> ["0","0","0","0","1"]
    """

    @staticmethod
    def is_symbol_valid(symbol) -> bool:
        return BitstringDataBlock.validate_data_symbol(symbol)

    def transform(self, data_block: BitstringDataBlock) -> BitsDataBlock:
        output_block = super().transform(data_block)
        return BitsDataBlock(output_block.data_list)


class BitsToBitstringTransformer(DataTransformer):
    """
    combines the bits into bitstrings
    eg: bit_width=2, ["0","0","0","1"] -> ["00", "01"]
    """

    def __init__(self, bit_width: int):
        self.bit_width = bit_width

    def transform(self, bits_block: BitsDataBlock) -> BitstringDataBlock:
        assert bits_block.size % self.bit_width == 0
        output_list: List = []
        _str = ""
        for ind, bit in enumerate(bits_block.data_list):
            if (ind != 0) and (ind % self.bit_width) == 0:
                output_list.append(_str)
                _str = ""
            assert BitsDataBlock.validate_data_symbol(bit)
            _str += str(bit)

        output_list.append(_str)

        return BitstringDataBlock(output_list)


class UintToBitstringTransformer(DataTransformer):
    """
    transforms uint8 data to bitstring.
    Each datapoint is represented using bit_width number of bits
    Eg: bit_width = 3, ["3", "5"] -> ["011", "101"]
    """

    def __init__(self, bit_width=None):
        self.bit_width = bit_width

    def transform(self, data_block: UintDataBlock):
        output_list: List[str] = []
        for symbol in data_block.data_list:
            assert UintDataBlock.validate_data_symbol(symbol)
            bitstring = uint_to_bitstring(symbol, bit_width=self.bit_width)
            output_list.append(bitstring)

        return BitstringDataBlock(output_list)


class BitstringToUintTransformer(DataTransformer):
    """
    transforms bitstring data (each symbol is a bitstring) to uint
    Eg: ["011", "101"] -> ["3", "5"]
    """

    def transform(self, data_block: BitstringDataBlock):

        output_list: List[str] = []
        for bitstring in data_block.data_list:
            assert BitstringDataBlock.validate_data_symbol(bitstring)
            uint_data = bitstring_to_uint(bitstring)
            output_list.append(uint_data)

        return UintDataBlock(output_list)


class LookupFuncTransformer(DataTransformer):
    """
    returns value by using the lookup function
    For example:
    If input block is [0,1,1,2], and lookup func is
    def func(x):
        return x*2
    then it will output [0, 2, 2, 4]
    """

    def __init__(self, lookup_func: Callable):
        self.lookup_func = lookup_func

    def transform(self, data_block: DataBlock):
        output_list = []
        for symbol in data_block.data_list:
            output_list.append(self.lookup_func(symbol))

        return DataBlock(output_list)


class LookupTableTransformer(LookupFuncTransformer):
    """
    returns value based on the lookup table.
    Can be implemented as a subclass of LookupFuncTransformer
    """

    def __init__(self, lookup_table: Dict):
        super().__init__(lookup_func=lambda x: lookup_table[x])


class CascadeTransformer(DataTransformer):
    """
    Runs multiple transformers in series
    """

    def __init__(self, transformer_list: List[DataTransformer]):
        self.transformer_list = transformer_list

    def transform(self, data_block: DataBlock):
        output_block = data_block
        for transformer in self.transformer_list:
            output_block = transformer.transform(output_block)
        return output_block


class BitsParserTransformer(DataTransformer):
    """
    Transformer which operates on BitsDataBlock and consumes bits.
    TODO: @tpulkit add more details
    """

    def __init__(self, parse_bits_func: Callable):

        # parse_bits_func needs to take in the data_block and a starting index
        # FIXME: @shubham this is ugly
        self.parse_bits_func = parse_bits_func

    def transform(self, data_block: BitsDataBlock):
        output_list = []
        assert isinstance(data_block, BitsDataBlock)

        # start parsing the bits datablock.
        # end the parsing when the end of the data_block has been reached
        start_ind = 0
        while start_ind < data_block.size:
            output_symbol, start_ind = self.parse_bits_func(data_block, start_ind)
            output_list.append(output_symbol)

        return DataBlock(output_list)
