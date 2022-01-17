import unittest
from compressors.huffman_coder import HuffmanCoder, HuffmanTree
import numpy as np

from core.data_stream import DataStream
from core.prob_dist import ProbabilityDist
from utils.test_utils import try_lossless_compression


class HuffmanCoderTest(unittest.TestCase):
    NUM_SAMPLES = 10000

    def _test_huffman_coding(self, prob_dist: ProbabilityDist):
        """
        1. Randomly generate data with the given distribution
        2. Construct Huffman coder using the given distribution
        3. Encode/Decode the stream
        """

        # generate random data
        data = np.random.choice(prob_dist.alphabet, self.NUM_SAMPLES, p=prob_dist.prob_list)

        # perform compression
        data_stream = DataStream(data)
        compressor = HuffmanCoder(prob_dist)
        is_lossless, output_len = try_lossless_compression(data_stream, compressor)

        avg_bits = output_len / self.NUM_SAMPLES
        return is_lossless, avg_bits

    def test_huffman_coding_dyadic(self):
        """
        On dyadic distributions Huffman coding should be perfectly equal to entropy
        """

        distributions = [
            ProbabilityDist({"A": 0.5, "B": 0.5}),
            ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.25}),
            ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.125, "D": 0.125}),
        ]
        print()
        for prob_dist in distributions:

            is_lossless, avg_bits = self._test_huffman_coding(prob_dist)
            assert is_lossless, "Lossless compression failed"
            np.testing.assert_almost_equal(
                avg_bits,
                prob_dist.entropy,
                decimal=2,
                err_msg="Huffman coding is not close to entropy",
            )
            print(f"Avg Bits: {avg_bits}, Entropy: {prob_dist.entropy}")
