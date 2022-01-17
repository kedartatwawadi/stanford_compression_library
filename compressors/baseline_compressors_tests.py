import unittest
from compressors.baseline_compressors import FixedBitwidthCompressor
from core.data_block import data_block
from utils.test_utils import try_lossless_compression


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

        is_lossless, codelen = try_lossless_compression(data_block, compressor)
        assert is_lossless

        # check if the length of the encoding was correct
        assert codelen == len(data_list) * 2
