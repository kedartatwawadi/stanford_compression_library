import abc
from core.data_stream import BitstringDataStream, DataStream, StringDataStream


class DataTransformer(abc.ABC):
    """
    a DataTransformer transform the input DataStream into an output DataStream
    """

    @abc.abstractmethod
    def transform(self, data_stream: DataStream) -> DataStream:
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
        return DataStream(output_list)

    @staticmethod
    def is_symbol_valid(symbol) -> bool:
        return StringDataStream.validate_data_symbol(symbol)


class BitstringToBits(SplitStringTransformer):
    """
    splits the input bitstring List into a list of bits
    """

    @staticmethod
    def is_symbol_valid(symbol) -> bool:
        return BitstringDataStream.validate_data_symbol(symbol)
