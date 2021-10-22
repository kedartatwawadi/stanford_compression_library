"""
Utility functions useful for testing
"""

from typing import Tuple
from core.data_stream import DataStream
from core.data_compressor import DataCompressor


def are_streams_equal(data_stream_1: DataStream, data_stream_2: DataStream):
    """
    return True is the streams are equal
    """
    if data_stream_1.size != data_stream_2.size:
        return False

    # check if the encoding/decoding was lossless
    for inp_symbol, out_symbol in zip(data_stream_1.data_list, data_stream_2.data_list):
        if inp_symbol != out_symbol:
            return False

    return True


def try_lossless_compression(
    data_stream: DataStream, compressor: DataCompressor
) -> Tuple[bool, int]:
    """
    Encodes the data_stream using data_compressor and returns True if the compression was lossless
    returns (True/False, size of the output stream)
    """
    try:
        # test encode
        output_bits_stream = compressor.encode(data_stream)

        # test decode
        decoded_stream = compressor.decode(output_bits_stream)

        # compare streams
        return are_streams_equal(data_stream, decoded_stream), output_bits_stream.size

    except:
        print("Error during Encoding/Decoding")
        return False, 0
