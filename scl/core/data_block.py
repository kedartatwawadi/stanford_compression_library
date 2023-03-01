from typing import List, Set
from scl.core.prob_dist import ProbabilityDist


class DataBlock:
    """
    wrapper around a list of symbols.

    The class is a wrapper around a list of symbols (self.data_list). The data_block is typically used
    to represent input to the data encoders (or output from data decoders)

    Some utility functions (useful generally for compression) implemented are:
    - size
    - alphabet
    - empirical_distribution
    - entropy
    """

    def __init__(self, data_list: List):
        self.data_list = data_list

    @property
    def size(self):
        return len(self.data_list)

    def get_alphabet(self) -> Set:
        """returns the set of unique symbols in the data_list

        Returns:
            Set: the alphabet
        """
        alphabet = set()
        for d in self.data_list:
            alphabet.add(d)
        return alphabet

    def get_counts(self, order=0):
        """returns a dictionary of counts for symbols in self.data_list

        Args:
            order (int, optional): (the order of k-mer for which counts need to be obtained) Defaults to 0.
            order 0-> symbol counts.. order 1-> tuple counts

        Raises:
            NotImplementedError: If order !=0, as it is not implemented yet

        Returns:
            dict: {symbol:count, ...}
        """

        if order != 0:
            raise NotImplementedError("[order != 0] counts not implemented")

        # get the alphabet
        alphabet = self.get_alphabet()

        # initialize the count dict
        count_dict = {a: 0 for a in alphabet}

        # populate the count dict
        for d in self.data_list:
            count_dict[d] += 1

        return count_dict

    def get_empirical_distribution(self, order=0) -> ProbabilityDist:
        """get empirical distribution for self.data_list of order 0

        Return the empirical distribution of the given order.
        for e.g if data_list = [A,B,A,A], order=0 -> returns {A:0.75, B:0.25}.
        if data_list = [A,B,A,A], order = 1 -> returns {AB:0.33, BA:0.33, AA:0.33}

        Args:
            order (int, optional): the order of counts. Defaults to 0.

        Raises:
            NotImplementedError: not implemented for order != 0 yet

        Returns:
            ProbabilityDist: the epirical distribution
        """

        if order != 0:
            raise NotImplementedError("[order != 0] empirical counts not implemented")

        # get counts
        counts_dict = self.get_counts()

        # compute the prob form the counts
        prob_dict = {}
        for symbol, count in counts_dict.items():
            prob_dict[symbol] = count / self.size

        return ProbabilityDist(prob_dict)

    def get_entropy(self, order=0):
        """Returns the entropy of the empirical_distribution of data_list

        returns the entropy of the empirical_distribution
        https://en.wikipedia.org/wiki/Entropy_(information_theory)
        """
        if order != 0:
            raise NotImplementedError("[order != 0] Entropy computation not implemented")

        prob_dist = self.get_empirical_distribution()
        return prob_dist.entropy


def test_data_block_basic_ops():
    """checks basic operations for a DataBlock"""
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
