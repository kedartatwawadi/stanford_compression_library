from typing import List
from core.util import compute_alphabet, compute_counts_dict
from core.prob_dist import ProbabilityDist
import unittest


class DataBlock:
    """
    wrapper around a list of symbols
    """

    def __init__(self, data_list: List):
        self.data_list = data_list

    @property
    def size(self):
        return len(self.data_list)

    def get_counts(self, order=0):
        """
        returns counts for each object
        """
        if order != 0:
            raise NotImplementedError("[order != 0] counts not implemented")

        return compute_counts_dict(self.data_list)

    def get_empirical_distribution(self, order=0) -> ProbabilityDist:
        """
        Computes the empirical distribution of the given order
        """
        if order != 0:
            raise NotImplementedError("[order != 0] Entropy computation not implemented")

        # get counts
        counts_dict = self.get_counts()

        # compute the prob form the counts
        prob_dict = {}
        for symbol, count in counts_dict.items():
            prob_dict[symbol] = count / self.size

        return ProbabilityDist(prob_dict)

    def get_entropy(self, order=0):
        """
        Computes the entropy of the given order
        """
        if order != 0:
            raise NotImplementedError("[order != 0] Entropy computation not implemented")

        prob_dist = self.get_empirical_distribution()
        return prob_dist.entropy

    def get_alphabet(self):
        return compute_alphabet(self.data_list)


class BitsDataBlock(DataBlock):
    pass


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
