import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class Symbol:
    """
    """
    id: Any = None
    prob: float = 0


class ProbabilityDist:
    """
    TODO: add description
    """

    def __init__(self, prob_dict=None):
        self._validate_prob_dist(prob_dict)

        self.symbol_list = []
        for id, prob in prob_dict.items():
            self.symbol_list.append(Symbol(id=id, prob=prob))

    @property
    def size(self):
        return len(self.symbol_list)
    
    # sorts the symbol_list according to the prob val
    def sort(self):
        self.symbol_list.sort(key=lambda x: x.prob)

    def add(self, id, prob):
        self.symbol_list.append(Symbol(id=id, prob=prob))

    def pop(self, ind=-1):
        return self.symbol_list.pop(ind)
    
    def get_symbol(self, ind):
        return self.symbol_list[ind]

    @property
    def entropy(self):
        entropy = 0
        for symbol in self.symbol_list:
            prob = symbol.prob
            entropy += -prob * np.log2(prob)
        return entropy

    @staticmethod
    def _validate_prob_dist(prob_dict):
        """
        checks if each value of the prob dist is non-negative,
        and the dist sums to 1
        """

        sum_of_probs = 0
        for _, prob in prob_dict.items():
            assert prob >= 0, "probabilities cannot be negative"
            sum_of_probs += prob

        # FIXME: check if this needs a tolerance range
        assert sum_of_probs == 1.0, "probabilities should sum to 1.0"




