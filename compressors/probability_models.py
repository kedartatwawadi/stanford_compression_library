"""This file contains probability models such as adaptive kth order etc. to
be used with arithmetic/range coders.

"""

import abc
import copy
from typing import List

import numpy as np

from core.prob_dist import Frequencies


class FreqModelBase(abc.ABC):
    """Base Freq Model

    The Arithmetic Entropy Coding (AEC) encoder can be thought of consisting of two parts:
    1. The probability model
    2. The "lossless coding" algorithm which uses these probabilities

    Note that the probabilities/frequencies coming from the probability model are fixed in the simplest Arithmetic coding
    version, but they can be modified as we parse each symbol.
    This class represents a generic "probability Model", but using frequencies (or counts), and hence the name FreqModel.
    Frequencies are used, mainly because floating point values can be unpredictable/uncertain on different platforms.

    Some typical examples of Freq models are:

    a) FixedFreqModel -> the probability model is fixed to the initially provided one and does not change
    b) AdaptiveIIDFreqModel -> starts with some initial probability distribution provided
        (the initial distribution is typically uniform)
        The Adaptive Model then updates the model based on counts of the symbols it sees.

    Args:
        freq_initial -> the frequencies used to initialize the model
        max_allowed_total_freq -> to limit the total_freq values of the frequency model
    """

    def __init__(self, freqs_initial: Frequencies, max_allowed_total_freq):
        # initialize the current frequencies using the initial freq.
        # NOTE: the deepcopy here is needed as we modify the frequency table internally
        # so, if it is used elsewhere externally, then it can cause unexpected issued
        self.freqs_current = copy.deepcopy(freqs_initial)
        self.max_allowed_total_freq = max_allowed_total_freq

    @abc.abstractmethod
    def update_model(self, s):
        """updates self.freqs

        Takes in as input the next symbol s and updates the
        probability distribution self.freqs (represented in terms of frequencies)
        appropriately. See examples below.
        """
        raise NotImplementedError  # update the probability model here


class FixedFreqModel(FreqModelBase):
    def update_model(self, s):
        """function to update the probability model

        In this case, we don't do anything as the freq model is fixed

        Args:
            s (Symbol): the next symbol
        """
        # nothing to do here as the freqs are always fixed
        pass


class AdaptiveIIDFreqModel(FreqModelBase):
    def update_model(self, s):
        """function to update the probability model

        - We start with uniform distribution on all symbols
        ```
        Freq = [A:1,B:1,C:1,D:1] for example.
        ```
        - Every time we see a symbol, we update the freq count by 1
        - Arithmetic coder requires the `total_freq` to remain below a certain value
        If the total_freq goes beyond, then we divide all freq by 2 (keeping minimum freq to 1)

        Args:
            s (Symbol): the next symbol
        """
        # updates the model based on the next symbol
        self.freqs_current.freq_dict[s] += 1

        # if total_freq goes beyond a certain value, divide by 2
        # NOTE: there can be different strategies here
        if self.freqs_current.total_freq >= self.max_allowed_total_freq:
            for s, f in self.freqs_current.freq_dict.items():
                self.freqs_current.freq_dict[s] = max(f // 2, 1)


class AdaptiveOrderKFreqModel(FreqModelBase):
    """kth order adaptive frequency model.

    Parameters:
        alphabet: the alphabet (provided as a list)
        k:        the order, k >= 0 (kth order means we use past k to predict next, k=0 means iid)
        max_allowed_total_freq: to limit the total_freq values of the frequency model
    """

    def __init__(self, alphabet: List, k: int, max_allowed_total_freq: int):
        assert k >= 0
        self.k = k
        # map alphabet to index from 0 to len(alphabet) so we can use with numpy array
        self.alphabet_to_idx = {alphabet[i]: i for i in range(len(alphabet))}
        # keep freq/counts of (k+1) tuples, initialize with all 1s (uniform)
        self.freqs_kplus1_tuple = np.ones([len(alphabet)] * (k + 1), dtype=int)
        self.max_allowed_total_freq = max_allowed_total_freq
        # keep track of past k symbols (i.e., alphabet index) seen. Initialize with all 0s.
        # Note that all zeros refers to the first element in the alphabet list. This is an
        # arbitrary choice made to simplify later processing rather than doing special case
        # for the first few symbols
        self.past_k = [0] * k

        self.alphabet = alphabet

    @property
    def freqs_current(self):
        """Calculate the current freqs. For order 0, we just give back the freqs. For k > 0,
        we use the past k symbols to pick out the corresponding frequencies for the (k+1)th.
        """
        if self.k > 0:
            # convert self.past_k to enable indexing
            # use np.ravel to convert to flat array
            freqs_given_context = np.ravel(self.freqs_kplus1_tuple[tuple(self.past_k)])
        else:
            freqs_given_context = self.freqs_kplus1_tuple
        # convert from list of frequencies to Frequencies object
        return Frequencies(dict(zip(self.alphabet, freqs_given_context)))

    def update_model(self, s):
        """function to update the probability model. This basically involves update the count
        for the most recently seen (k+1) tuple.

        - Arithmetic coder requires the `total_freq` to remain below a certain value
        If the total_freq goes beyond, then we divide all freq by 2 (keeping minimum freq to 1)

        Args:
            s (Symbol): the next symbol
        """
        # updates the model based on the new symbol
        # index self.freqs_kplus1_tuple using (past_k, s) [need to map s to index]
        current_tuple = (*self.past_k, self.alphabet_to_idx[s])
        self.freqs_kplus1_tuple[current_tuple] += 1

        # if k > 0, update past_k list
        if self.k > 0:
            self.past_k = self.past_k[1:] + [self.alphabet_to_idx[s]]

        # if total_freq goes beyond a certain value, divide by 2
        # NOTE: there can be different strategies here
        # NOTE: we only need the frequencies for each (k+1) tuple to
        # sum to less than max_allowed_total_freq
        if np.sum(self.freqs_kplus1_tuple[current_tuple]) >= self.max_allowed_total_freq:
            self.freqs_kplus1_tuple[current_tuple] = np.max(self.freqs_kplus1_tuple[current_tuple] // 2, 1)
