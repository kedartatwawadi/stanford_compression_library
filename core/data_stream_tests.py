import unittest
from core.data_stream import DataStream


class DataStreamTest(unittest.TestCase):
    """
    checks basic operations for a DataStream
    FIXME: improve these tests
    """

    def test_data_stream_basic_ops(self):
        data_list = [0, 1, 0, 0, 1, 1]

        # create data stream object
        data_stream = DataStream(data_list)

        # check size
        assert data_stream.size == 6

        # check counts
        counts_dict = data_stream.get_counts(order=0)
        assert counts_dict[0] == 3

        # check empirical dist
        prob_dist = data_stream.get_empirical_distribution(order=0)
        assert prob_dist.prob_dict[0] == 0.5

        # check entropy
        entropy = data_stream.get_entropy(order=0)
        assert entropy == 1.0
