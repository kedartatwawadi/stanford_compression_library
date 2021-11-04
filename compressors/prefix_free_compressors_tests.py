import unittest
from compressors.prefix_free_compressors import UniversalUintCompressor
from core.data_stream import UintDataStream


class UintUniversalCompressorTest(unittest.TestCase):
    """
    FIXME: improve these tests
    """

    def test_encode_decode(self):
        compressor = UniversalUintCompressor()

        # create some sample data
        data_list = [0, 0, 1, 3, 4, 100]
        data_stream = UintDataStream(data_list)

        # test encode
        output_bits_stream = compressor.encode(data_stream)

        # test decode
        decoded_stream = compressor.decode(output_bits_stream)

        # check if the encoding/decoding was lossless
        for inp_symbol, out_symbol in zip(data_stream.data_list, decoded_stream.data_list):
            assert inp_symbol == out_symbol
