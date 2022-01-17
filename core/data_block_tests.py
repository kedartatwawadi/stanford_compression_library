import unittest
from core.data_block import DataBlock, StringDataBlock, BitstringDataBlock


class DataBlockTest(unittest.TestCase):
    """
    checks basic operations for a DataBlock
    FIXME: improve these tests
    """

    def test_data_block_basic_ops(self):
        data_list = [0, 1, 0, 0, 1, 1]

        # create data block object
        data_block = DataBlock(data_list)

        # check size
        assert data_block.size == 6

        # check counts
        counts_dict = data_block.get_counts(order=0)
        assert counts_dict[0] == 3

        # check empirical dist
        prob_dist = data_block.get_empirical_distribution(order=0)
        assert prob_dist.prob_dict[0] == 0.5

        # check entropy
        entropy = data_block.get_entropy(order=0)
        assert entropy == 1.0


class StringDataBlockTest(unittest.TestCase):
    """
    checks the validation func of the StringDataBlock class
    """

    def test_validate_func_on_valid_input(self):
        data_list = ["0", "11", "001", "0011"]
        for d in data_list:
            assert StringDataBlock.validate_data_symbol(d)

    @unittest.expectedFailure
    def test_validate_func_on_invalid_input(self):
        data_list = [0, 1, 0, 0, 1, 1]
        for d in data_list:
            assert StringDataBlock.validate_data_symbol(d)


class BitstringDataBlockTest(unittest.TestCase):
    """
    checks the validation func of the StringDataBlock class
    """

    def test_validate_func_on_valid_input(self):
        data_list = ["0", "11", "001", "0011"]
        for d in data_list:
            assert StringDataBlock.validate_data_symbol(d)

    @unittest.expectedFailure
    def test_validate_func_on_invalid_input(self):
        data_list = ["0", "A1", "AA1", "AA110"]
        for d in data_list:
            assert BitstringDataBlock.validate_data_symbol(d)
