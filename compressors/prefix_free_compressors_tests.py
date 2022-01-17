import unittest
from compressors.prefix_free_compressors import UniversalUintCompressor
from core.data_block import UintDataBlock


class UintUniversalCompressorTest(unittest.TestCase):
    """
    FIXME: @tpulkit -> improve these tests
    """

    def test_encode_decode(self):
        compressor = UniversalUintCompressor()

        # create some sample data
        data_list = [0, 0, 1, 3, 4, 100]
        data_block = UintDataBlock(data_list)

        # test encode
        output_bits_block = compressor.encode(data_block)

        # test decode
        decoded_block = compressor.decode(output_bits_block)

        # check if the encoding/decoding was lossless
        for inp_symbol, out_symbol in zip(data_block.data_list, decoded_block.data_list):
            assert inp_symbol == out_symbol
