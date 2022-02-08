from typing import List, Set
from dataclasses import dataclass
from core.util import compute_alphabet, compute_counts_dict
from core.prob_dist import ProbabilityDist


class DataBlock:
    """
    FIXME: for now data is only presented as a list of symbols
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

    @classmethod
    def char_to_symbol(cls, c):
        return c


class UintDataBlock(DataBlock):
    """
    block consisting of unsigned integers
    """

    @staticmethod
    def validate_data_symbol(symbol) -> bool:
        """
        validates that the symbol is of type unsigned int
        """
        if not isinstance(symbol, int):
            return False
        return symbol >= 0

    @classmethod
    def char_to_symbol(cls, c):
        return int(c)


class BitsDataBlock(DataBlock):
    """
    block consisting of bits. either ("0" or "1")
    or (0,1)
    """

    @staticmethod
    def validate_data_symbol(symbol) -> bool:
        """
        validates that the symbol is of type str
        """

        if isinstance(symbol, str):
            return (symbol == "0") or (symbol == "1")
        elif isinstance(symbol, int):
            return (symbol == 0) or (symbol == 1)
        else:
            return False
