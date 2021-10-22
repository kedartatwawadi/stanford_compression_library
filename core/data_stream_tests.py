import unittest
from core.data_stream import DataStream, StringDataStream, BitstringDataStream


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
        prob_dist.prob_dict[0] == 0.5

        # check entropy
        entropy = data_stream.get_entropy(order=0)
        assert entropy == 1.0


class StringDataStreamTest(unittest.TestCase):
    """
    checks the validation func of the StringDataStream class
    """

    def test_validate_func_on_valid_input(self):
        data_list = ["0", "11", "001", "0011"]
        for d in data_list:
            assert StringDataStream.validate_data_symbol(d)

    @unittest.expectedFailure
    def test_validate_func_on_invalid_input(self):
        data_list = [0, 1, 0, 0, 1, 1]
        for d in data_list:
            assert StringDataStream.validate_data_symbol(d)


class BitstringDataStreamTest(unittest.TestCase):
    """
    checks the validation func of the StringDataStream class
    """

    def test_validate_func_on_valid_input(self):
        data_list = ["0", "11", "001", "0011"]
        for d in data_list:
            assert StringDataStream.validate_data_symbol(d)

    @unittest.expectedFailure
    def test_validate_func_on_invalid_input(self):
        data_list = ["0", "A1", "AA1", "AA110"]
        for d in data_list:
            assert BitstringDataStream.validate_data_symbol(d)
