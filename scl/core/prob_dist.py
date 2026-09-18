import numpy as np
import unittest
from operator import index


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
        for s, curr_symbol_prob in sorted_PD.prob_dict.items():
            assert curr_symbol_prob <= prev_symbol_prob
            prev_symbol_prob = curr_symbol_prob

        # assert the elements of the new sorted dict is same as pre-sorting
        assert sorted_PD.prob_dict == dist

    def test_properties_follow_probability_mutations(self):
        probabilities = {"A": 0.5, "B": 0.5}
        dist = ProbabilityDist(probabilities)
        self.assertEqual(dist.entropy, 1.0)
        self.assertEqual(dist.cumulative_prob_dict, {"A": 0, "B": 0.5})

        probabilities.update({"A": 0.75, "B": 0.25})
        self.assertAlmostEqual(dist.entropy, 0.8112781244591328)
        self.assertEqual(dist.cumulative_prob_dict, {"A": 0, "B": 0.75})

        cumulative = dist.cumulative_prob_dict
        cumulative["B"] = 99
        self.assertEqual(dist.cumulative_prob_dict, {"A": 0, "B": 0.75})


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
    The input mapping is copied into Python integers; mutate freq_dict to update this table.
    """

    def __init__(self, freq_dict=None):
        if freq_dict is None:
            raise ValueError("freq_dict must be provided")
        # NOTE: We use the fact that since python 3.6, dictionaries in python are
        # also OrderedDicts. https://realpython.com/python-ordereddict/
        self.freq_dict = {symbol: index(freq) for symbol, freq in freq_dict.items()}
        self._validate_freq_dist(self.freq_dict)

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
        return sum(index(freq) for freq in self.freq_dict.values())

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
            _sum += index(p)
        return cum_freq_dict

    def frequency(self, symbol):
        return index(self.freq_dict[symbol])

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
        """Check that every frequency is a positive Python integer."""
        for _, freq in freq_dict.items():
            if not isinstance(freq, int):
                raise TypeError("frequency must be an integer")
            if freq <= 0:
                raise ValueError("frequency cannot be negative or 0")


class FrequenciesTest(unittest.TestCase):
    def test_numpy_counts_use_exact_python_arithmetic(self):
        largest = np.uint64(2**64 - 1)
        freqs = Frequencies({"A": largest, "B": largest, "C": np.int64(1)})
        self.assertTrue(all(type(freq) is int for freq in freqs.freq_list))
        self.assertEqual(freqs.total_freq, 2**65 - 1)
        self.assertIs(type(freqs.total_freq), int)
        self.assertEqual(freqs.cumulative_freq_dict, {"A": 0, "B": 2**64 - 1, "C": 2**65 - 2})
        self.assertTrue(all(type(freq) is int for freq in freqs.cumulative_freq_dict.values()))

    def test_frequency_dictionary_remains_mutable(self):
        freqs = Frequencies({"A": np.int64(2), "B": 1})
        freqs.freq_dict["A"] += 1
        self.assertEqual(freqs.frequency("A"), 3)
        self.assertEqual(freqs.total_freq, 4)
        self.assertEqual(freqs.cumulative_freq_dict, {"A": 0, "B": 3})

        freqs.freq_dict["A"] = np.uint64(2**64 - 1)
        self.assertIs(type(freqs.frequency("A")), int)
        self.assertEqual(freqs.total_freq, 2**64)
        self.assertIs(type(freqs.total_freq), int)
        self.assertIs(type(freqs.cumulative_freq_dict["B"]), int)

    def test_invalid_frequencies_fail_during_construction(self):
        for frequency in [0, -1, np.int64(-1)]:
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                Frequencies({"A": frequency})
        for frequency in [1.5, np.float64(2), "1", None]:
            with self.subTest(frequency=frequency), self.assertRaises(TypeError):
                Frequencies({"A": frequency})
        with self.assertRaises(ValueError):
            Frequencies()
