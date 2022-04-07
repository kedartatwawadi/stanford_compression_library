import numpy as np
import unittest


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

    @property
    def cumulative_prob_dict(self):
        """return a list of sum of probabilities of symbols preceeding symbol"""
        cum_prob_dict = {}
        _sum = 0
        for a, p in self.prob_dict.items():
            cum_prob_dict[a] = _sum
            _sum += p
        return cum_prob_dict

    @property
    def entropy(self):
        entropy = 0
        for _, prob in self.prob_dict.items():
            entropy += -prob * np.log2(prob)
        return entropy

    def probability(self, symbol):
        return self.prob_dict[symbol]

    def log_probability(self, symbol):
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
