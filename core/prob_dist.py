import numpy as np


class ProbabilityDist:
    """
    TODO: add description
    """

    def __init__(self, prob_dict):
        self.prob_dict = prob_dict
        self._validate_prob_dist(self.prob_dict)

    @property
    def entropy(self):
        entropy = 0
        for _, prob in self.prob_dict.items():
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
