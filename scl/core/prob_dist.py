import numpy as np
import unittest
from scl.utils.misc_utils import cache


class ProbabilityDist:
    """
    Wrapper around a probability dict
    """

    def __init__(self, prob_dict=None):
        self._validate_prob_dist(prob_dict)

        # NOTE: We use the fact that since python 3.6, dictionaries in python are
        # also OrderedDicts. https://realpython.com/python-ordereddict/
        self.prob_dict = prob_dict

    def __repr__(self):
        return f"ProbabilityDist({self.prob_dict.__repr__()}"

    @property
    def size(self):
        return len(self.prob_dict)

    @property
    def alphabet(self):
        return list(self.prob_dict)

    @property
    def prob_list(self):
        return [self.prob_dict[s] for s in self.alphabet]

    @classmethod
    def get_sorted_prob_dist(cls, prob_dict, descending=False):
        """
        Returns ProbabilityDist class object with sorted probabilities.
        By default, returns Probabilities in increasing order (descending=False), i.e.,
        p1 <= p2 <= .... <= pn (python-default)
        """
        return cls(dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=descending)))

    @classmethod
    def normalize_prob_dict(cls, prob_dict):
        """
        normalizes dict -> dict_norm so that the sum of values is 1
        wraps dict_norm as a ProbabilityDist
        """
        sum_p = sum(prob_dict.values())
        return cls({a: b / sum_p for a, b in prob_dict.items()})

    @property
    @cache
    def cumulative_prob_dict(self):
        """return a list of sum of probabilities of symbols preceeding symbol"""
        cum_prob_dict = {}
        _sum = 0
        for a, p in self.prob_dict.items():
            cum_prob_dict[a] = _sum
            _sum += p
        return cum_prob_dict

    @property
    @cache
    def entropy(self):
        entropy = 0
        for _, prob in self.prob_dict.items():
            entropy += -prob * np.log2(prob)
        return entropy

    def probability(self, symbol):
        return self.prob_dict[symbol]

    def neg_log_probability(self, symbol):
        return -np.log2(self.probability(symbol))

    @staticmethod
    def _validate_prob_dist(prob_dict):
        """
        checks if each value of the prob dist is non-negative,
        and the dist sums to 1
        """

        sum_of_probs = 0
        for _, prob in prob_dict.items():
            assert prob >= 1e-6, "probabilities negative or too small cause stability issues"
            sum_of_probs += prob

        # FIXME: check if this needs a tolerance range
        if abs(sum_of_probs - 1.0) > 1e-8:
            raise ValueError("probabilities do not sum to 1")


class ProbabilityDistTest(unittest.TestCase):
    def test_creation_entropy(self):
        """
        checks if the creation and validity checks are passing for valid distribution
        """

        # create valid distributions for testing
        fair_coin_dist = ProbabilityDist({"H": 0.5, "T": 0.5})
        assert fair_coin_dist.entropy == 1.0

        dyadic_dist = ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.25})
        assert dyadic_dist.entropy == 1.5

    @unittest.expectedFailure
    def test_validation_failure(self):
        """
        test if init fails for incorrect distributions
        """

        dist_1 = ProbabilityDist({"H": 0.5, "T": 0.4})

    def test_prob_creation_and_validation(self):
        """Test if validation works fine

        NOTE: Test added to check if issue #21 was resolved
        """
        alphabet = list(range(10))
        dist = {i: 1 / 10 for i in alphabet}

        # check if this works
        _ = ProbabilityDist(dist)

    def test_sorted_prob_dist(self):
        """
        checks if sorting works as expected and doesn't change the dict.
        """
        alphabet = list(range(10))
        dist = {i: (i + 1) / 55 for i in alphabet}

        sorted_PD = ProbabilityDist.get_sorted_prob_dist(dist, descending=True)
        # initialize to max prob
        prev_symbol_prob = 1
        for (s, curr_symbol_prob) in sorted_PD.prob_dict.items():
            assert curr_symbol_prob <= prev_symbol_prob
            prev_symbol_prob = curr_symbol_prob

        # assert the elements of the new sorted dict is same as pre-sorting
        assert sorted_PD.prob_dict == dist


def get_avg_neg_log_prob(prob_dist: ProbabilityDist, data_block) -> float:
    """computes the average neg_log_probability of the input data_block given the probability distribution
    prob_dist. This roughly is equal to what an optimal compressor designed for distribution
    prob_dist can achieve for the input data_block

    Args:
        prob_dist (ProbabilityDist): specified probability distribution used to compute neg_log_probability
        data_block (DataBlock): input for which avg log probability needs to be computed
    """

    # get avg log probability for the input
    log_prob = 0
    for s in data_block.data_list:
        log_prob += prob_dist.neg_log_probability(s)
    avg_log_prob = log_prob / data_block.size
    return avg_log_prob


class Frequencies:
    """
    Wrapper around a frequency dict
    NOTE: Frequencies is a typical way to represent probability distributions using integers
    """

    def __init__(self, freq_dict=None):

        # NOTE: We use the fact that since python 3.6, dictionaries in python are
        # also OrderedDicts. https://realpython.com/python-ordereddict/
        self.freq_dict = freq_dict

    def __repr__(self):
        return f"Frequencies({self.freq_dict.__repr__()}"

    @property
    def size(self):
        return len(self.freq_dict)

    @property
    def alphabet(self):
        return list(self.freq_dict)

    @property
    def freq_list(self):
        return [self.freq_dict[s] for s in self.alphabet]

    @property
    def total_freq(self) -> int:
        """returns the sum of all the frequencies"""
        return np.sum(self.freq_list)

    @property
    def cumulative_freq_dict(self) -> dict:
        """return a list of sum of probabilities of symbols preceeding symbol
        for example: freq_dict = {A: 7,B: 1,C: 3}
        cumulative_freq_dict = {A: 0, B: 7, C: 8}

        """
        cum_freq_dict = {}
        _sum = 0
        for a, p in self.freq_dict.items():
            cum_freq_dict[a] = _sum
            _sum += p
        return cum_freq_dict

    def frequency(self, symbol):
        return self.freq_dict[symbol]

    def get_prob_dist(self) -> ProbabilityDist:
        """_summary_

        Returns:
            _type_: _description_
        """
        prob_dict = {}
        for s, f in self.freq_dict.items():
            prob_dict[s] = f / self.total_freq
        return ProbabilityDist(prob_dict)

    @staticmethod
    def _validate_freq_dist(freq_dict):
        """
        checks if each value of the prob dist is non-negative,
        and the dist sums to 1
        """

        for _, freq in freq_dict.items():
            assert freq > 0, "frequency cannot be negative or 0"
            assert isinstance(freq, int)
