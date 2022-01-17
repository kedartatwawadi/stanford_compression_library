import unittest
from compressors.prefix_free_compressors import (
    GolombUintCompressor,
    UniversalUintCompressor,
)
from core.data_stream import UintDataStream


class UintUniversalCompressorTest(unittest.TestCase):
    """
    FIXME: @tpulkit -> improve these tests
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


class GolombUintCompressorTest(unittest.TestCase):
    """
    FIXME: @tpulkit -> improve these tests
    """

    def _test_encode_decode_helper(M, data_list, expected_output_bitstring):
        compressor = GolombUintCompressor(M)

        # sample data
        data_stream = UintDataStream(data_list)

        # test encode
        output_bits_stream = compressor.encode(data_stream)

        assert "".join(output_bits_stream.data_list) == expected_output_bitstring

        # test decode
        decoded_stream = compressor.decode(output_bits_stream)

        # check if the encoding/decoding was lossless
        for inp_symbol, out_symbol in zip(data_stream.data_list, decoded_stream.data_list):
            assert inp_symbol == out_symbol

    def test_encode_decode(self):
        # first test with M power of 2
        M = 4  # so b = 2 and cutoff = 4 (cutoff can be ignored for M power of 2 which is just Rice code)
        data_list = [0, 1, 4, 102]
        expected_output_bitstring = "000" + "001" + "1000" + "1" * 25 + "0" + "10"
        GolombUintCompressorTest._test_encode_decode_helper(M, data_list, expected_output_bitstring)

        # test with M not power of 2
        M = 10  # so b = 3 and cutoff = 6
        data_list = [2, 7, 26, 102]
        expected_output_bitstring = "0010" + "01101" + "1101100" + "11111111110010"
        GolombUintCompressorTest._test_encode_decode_helper(M, data_list, expected_output_bitstring)
