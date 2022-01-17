import unittest
from compressors.prefix_free_compressors import UniversalUintCompressor
from core.data_stream import UintDataStream
from utils.test_utils import try_lossless_compression


class UintUniversalCompressorTest(unittest.TestCase):
    """
    FIXME: @tpulkit -> improve these tests
    """

    def test_encode_decode(self):
        compressor = UniversalUintCompressor()

        # create some sample data
        data_list = [0, 0, 1, 3, 4, 100]
        data_stream = UintDataStream(data_list)

        is_lossless, codelen = try_lossless_compression(data_stream, compressor)
        assert is_lossless
