import unittest
from compressors.baseline_compressors import FixedBitwidthCompressor
from core.data_stream import DataStream


class FixedBitwidthCompressorTest(unittest.TestCase):
    """
    checks basic operations for a DataStream
    FIXME: improve these tests
    """

    def test_encode_decode(self):
        compressor = FixedBitwidthCompressor()

        # create some sample data
        data_list = ["A", "B", "C", "C", "A", "C"]
        data_stream = DataStream(data_list)

        # test encode
        output_bits_stream = compressor.encode(data_stream)

        # test decode
        decoded_stream = compressor.decode(output_bits_stream)

        # check if the encoding/decoding was lossless
        for inp_symbol, out_symbol in zip(data_stream.data_list, decoded_stream.data_list):
            assert inp_symbol == out_symbol

        # check if the length of the encoding was correct
        assert output_bits_stream.size == 12
