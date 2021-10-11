import abc
from core.data_stream import (
    BitsDataStream,
    BitstringDataStream,
    DataStream,
    StringDataStream,
    UintDataStream,
)
from typing import Callable, List, Dict

from core.util import bitstring_to_uint, uint_to_bitstring


class DataTransformer(abc.ABC):
    """
    a DataTransformer transform the input DataStream into an output DataStream
    """

    @abc.abstractmethod
    def transform(self, data_stream: DataStream) -> DataStream:
        """
        NOTE: the transform function of every DataTransformer takes only the data_stream as input
        """
        return None


class IdentityTransformer(DataTransformer):
    """
    returns the data stream
    """

    def transform(self, data_stream: DataStream) -> DataStream:
        return data_stream


class SplitStringTransformer(DataTransformer):
    """
    assumes input symbol is a string, and splits the string into individual chars
    For eg:
    ["aa", "aab"] -> ["a","a","a","a","b]
    """

    def transform(self, data_stream: StringDataStream) -> DataStream:
        output_list = []
        for symbol in data_stream.data_list:
            # check if the symbol is valid
            assert self.is_symbol_valid(symbol)

            output_list += [c for c in symbol]
        return StringDataStream(output_list)

    @staticmethod
    def is_symbol_valid(symbol) -> bool:
        return StringDataStream.validate_data_symbol(symbol)


class BitstringToBitsTransformer(SplitStringTransformer):
    """
    splits the input bitstring List into a list of bits
    """

    @staticmethod
    def is_symbol_valid(symbol) -> bool:
        return BitstringDataStream.validate_data_symbol(symbol)

    def transform(self, data_stream: BitstringDataStream) -> BitsDataStream:
        output_stream = super().transform(data_stream)
        return BitsDataStream(output_stream.data_list)


class BitsToBitstringTransformer(DataTransformer):
    """
    splits the input bitstring List into a list of bits
    """

    def __init__(self, bit_width: int = None):
        self.bit_width = bit_width

    def transform(self, bits_stream: BitsDataStream) -> BitstringDataStream:
        assert bits_stream.size % self.bit_width == 0
        output_list: List = []
        _str = ""
        for ind, bit in enumerate(bits_stream.data_list):
            if (ind != 0) and (ind % self.bit_width) == 0:
                output_list.append(_str)
                _str = ""
            assert BitsDataStream.validate_data_symbol(bit)
            _str += str(bit)

        output_list.append(_str)

        return BitstringDataStream(output_list)


class UintToBitstringTransformer(DataTransformer):
    """
    transforms uint8 data to bitstring.
    Each datapoint is represented using bit_width number of bits
    Eg:
    """

    def __init__(self, bit_width=None):
        self.bit_width = bit_width

    def transform(self, data_stream: UintDataStream):
        output_list: List[str] = []
        for symbol in data_stream.data_list:
            assert UintDataStream.validate_data_symbol(symbol)
            bitstring = uint_to_bitstring(symbol, bit_width=self.bit_width)
            output_list.append(bitstring)

        return BitstringDataStream(output_list)


class BitstringToUintTransformer(DataTransformer):
    """
    transforms bitstring data (each symbol is a bitstring) to uint
    Eg:
    """

    def transform(self, data_stream: BitstringDataStream):

        output_list: List[str] = []
        for bitstring in data_stream.data_list:
            assert BitstringDataStream.validate_data_symbol(bitstring)
            uint_data = bitstring_to_uint(bitstring)
            output_list.append(uint_data)

        return UintDataStream(output_list)


class LookupFuncTransformer(DataTransformer):
    """
    returns value by using the lookup function
    For example:
    If input stream is [0,1,1,2], and lookup func is
    def func(x):
        return x*2
    then it will output [0, 2, 2, 4]
    """

    def __init__(self, lookup_func: Callable):
        self.lookup_func = lookup_func

    def transform(self, data_stream: DataStream):
        output_list = []
        for symbol in data_stream.data_list:
            output_list.append(self.lookup_func(symbol))

        return DataStream(output_list)


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

    def transform(self, data_stream: DataStream):
        output_stream = data_stream
        for transformer in self.transformer_list:
            output_stream = transformer.transform(output_stream)
        return output_stream
