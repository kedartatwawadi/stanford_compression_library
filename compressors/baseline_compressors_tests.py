import unittest
from compressors.baseline_compressors import FixedBitwidthCompressor
from core.data_block import DataBlock


class FixedBitwidthCompressorTest(unittest.TestCase):
    """
    checks basic operations for a DataBlock
    FIXME: improve these tests
    """

    def test_encode_decode(self):
        compressor = FixedBitwidthCompressor()

        # create some sample data
        data_list = ["A", "B", "C", "C", "A", "C"]
        data_block = DataBlock(data_list)

        # test encode
        output_bits_block = compressor.encode(data_block)

        # test decode
        decoded_block = compressor.decode(output_bits_block)

        # check if the encoding/decoding was lossless
        for inp_symbol, out_symbol in zip(data_block.data_list, decoded_block.data_list):
            assert inp_symbol == out_symbol

        # check if the length of the encoding was correct
        assert output_bits_block.size == 12
