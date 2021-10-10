from typing import List, Set
from dataclasses import dataclass
from core.util import compute_alphabet, compute_counts_dict
from core.prob_dist import ProbabilityDist


@dataclass
class DataStream:
    """
    FIXME: for now data is only presented as a list of symbols
    """

    data_list: List = None
    alphabet: Set = None

    def set_alphabet(self):
        self.alphabet = compute_alphabet(self.data_list)

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
