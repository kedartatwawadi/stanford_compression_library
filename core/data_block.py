from typing import List, Set
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
        :param order: FIXME: I am assuming order here is the entropy order. Shouldn't it be defaulted to 1?
        :return: counts for each object
        """
        if order != 0:
            raise NotImplementedError("[order != 0] counts not implemented")

        return compute_counts_dict(self.data_list)

    def get_empirical_distribution(self, order=0) -> ProbabilityDist:
        """
        :param order: FIXME
        :return: Computes the empirical distribution of the given order

        """
        if order != 0:
            raise NotImplementedError("[order != 0] Entropy computation not implemented")

        # get counts
        counts_dict = self.get_counts()

        # compute the prob form the counts
        prob_dict = {
            symbol: count / self.size for symbol, count in counts_dict.items()
        }

        return ProbabilityDist(prob_dict)

    def get_entropy(self, order=0):
        """
        :param order: FIXME
        :return: Computes the entropy of the given order
        """

        if order != 0:
            raise NotImplementedError("[order != 0] Entropy computation not implemented")

        prob_dist = self.get_empirical_distribution()
        return prob_dist.entropy

    def get_alphabet(self) -> Set:
        """
        :return: set of alphabets from a list of data
        """
        return compute_alphabet(self.data_list)


class StringDataBlock(DataBlock):
    """
    DataBlock for which each element of the data_list is a str
    For eg: ["0", "1"], ["A", "AAB", "BCE"]
    """

    @staticmethod
    def validate_data_symbol(symbol) -> bool:
        """
        validates that the symbol is of type str
        """
        return isinstance(symbol, str)


class BitstringDataBlock(StringDataBlock):
    """
    DataBlock for which each element of the data_list is a string containing only binary "0" or "1" values
    For eg: ["0", "1", "0"], but not ["A", "AAB", "BCE"]
    """

    @staticmethod
    def validate_data_symbol(bitstring) -> bool:
        """
        validates that the symbol is either "0" or "1"
        """
        # validate if input symbol is a string
        is_str = isinstance(bitstring, str)
        if is_str:
            # validate if input symbol string contains only 0,1

            # FIXME: Why do we need bitstring_list, since this is inherited class shoudn't it be list by default?
            bitstring_list = list(bitstring)
            alphabet = compute_alphabet(bitstring_list)

            binary_alphabet = {"0", "1"}
            return alphabet.issubset(binary_alphabet)
        else:
            return False


class UintDataBlock(DataBlock):
    """
    block consisting of unsigned integers
    """

    @staticmethod
    def validate_data_symbol(symbol) -> bool:
        """
        validates that the symbol is of type unsigned int
        """
        return False if not isinstance(symbol, int) else symbol >= 0


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
            return symbol in ["0", "1"]
        elif isinstance(symbol, int):
            return symbol in [0, 1]
        else:
            return False
