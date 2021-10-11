from typing import Callable, List
from core.data_stream import BitsDataStream, DataStream
from core.data_transformer import DataTransformer


class BitsParserTransformer(DataTransformer):
    """
    Transformer which operates on BitsDataStream and consumes bits.
    TODO: add more details
    """

    def __init__(self, parse_bits_func: Callable):

        # parse_bits_func needs to take in
        self.parse_bits_func = parse_bits_func

    def transform(self, data_stream: BitsDataStream):
        output_list = []
        assert isinstance(data_stream, BitsDataStream)

        # start parsing the bits datastream.
        # end the parsing when the end of the data_stream has been reached
        start_ind = 0
        while start_ind < data_stream.size:
            output_symbol, start_ind = self.parse_bits_func(data_stream, start_ind)
            output_list.append(output_symbol)

        return DataStream(output_list)
