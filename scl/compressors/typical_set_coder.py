"""Typical set coder

For a given iid source $X_1,X_2,...$ i.i.d. $\sim X$ (from alphabet $\mathcal{X}$), 
and for given values of $n > 0$ and $\epsilon > 0$, the typical set is defined as 
$$A_{\epsilon}^{(n)} = \left\{x^n | \left|-\frac{1}{n}\log P(x^n) - H(X)\right| \leq \epsilon\right\}$$

This code works by dividing the input into chunks of size n, and then encodes each chunk
as follows:
- if $x^n \in A_{\epsilon}^{(n)}$, encode as a $0$ followed by a 
$\lceil\log_2|A_{\epsilon}^{(n)}|\rceil$ bit representation of the index of $x^n$ in the typical set
- otherwise, encode as $1$ followed by a $\lceil\log_2|\mathcal{X}|\rceil$ representation 
of the index of $x^n$ in $\mathcal{X}^n$

Note: a single data_block contains several of these n-length chunks, and in literature often
these chunks are referred to as blocks (as in block coding).
"""

from dataclasses import dataclass
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.core.prob_dist import ProbabilityDist
import numpy as np
import itertools
import random


def compute_normalized_negative_log_prob_chunk(chunk, prob_dist: ProbabilityDist):
    """Compute the normalized log probability for a chunk"""
    log_prob = 0
    for symbol in chunk:
        log_prob += prob_dist.neg_log_probability(symbol)
    return log_prob / len(chunk)


def is_typical(chunk, prob_dist: ProbabilityDist, eps):
    """Check if a chunk is typical given a distribution and value of epsilon"""
    return (
        np.abs(compute_normalized_negative_log_prob_chunk(chunk, prob_dist) - prob_dist.entropy)
        <= eps
    )


@dataclass
class TypicalSetCoderParams:
    """
    Parameters for the typical set encoder/decoder.
    """

    # the chunk size
    n: int

    # the epsilon used in definition of typical set.
    # An n-length sequence is typical if its normalized negative
    # log probability is within eps of the entropy.
    eps: float

    # the probability distribution used to define typical elements
    prob_dist: ProbabilityDist

    def __init__(self, n: int, eps: float, prob_dist: ProbabilityDist):
        """Initialize the typical set coder params"""
        self.n = n
        self.eps = eps
        self.prob_dist = prob_dist


def generate_typical_set_coder_lookup_tables(params: TypicalSetCoderParams):
    """generate the dicts used for encoding/decoding of typical set code"""
    index_typical = {}  # mapping from input typical sequence to its index in the typical set
    index_overall = {}  # mapping from an input sequence to its index in \mathcal{X}^n set
    # both of these map tuples of length n to non-negative integers

    num_typical_seen = 0  # use for indexing typical
    num_overall_seen = 0  # use for indexing overall

    for l in itertools.product(params.prob_dist.alphabet, repeat=params.n):
        if is_typical(l, params.prob_dist, params.eps):
            index_typical[tuple(l)] = num_typical_seen  # need to cast to tuple to use as dict key
            num_typical_seen += 1
        index_overall[tuple(l)] = num_overall_seen
        num_overall_seen += 1

    return index_typical, index_overall


class TypicalSetEncoder(DataEncoder):
    """Typical set encoder working on chunks of length n. See module documentation for more details."""

    def __init__(self, params: TypicalSetCoderParams) -> None:
        super().__init__()
        self.params = params
        self.index_typical, self.index_overall = generate_typical_set_coder_lookup_tables(
            self.params
        )
        # compute lengths of the encoding of the index for typical and non-typical cases
        if len(self.index_typical) == 0:
            self.index_bitlen_typical = None  # empty typical set, so can never encounter
        else:
            self.index_bitlen_typical = int(np.ceil(np.log2(len(self.index_typical))))
        self.index_bitlen_overall = int(np.ceil(np.log2(len(self.index_overall))))

    def encode_block(self, data_block: DataBlock) -> BitArray:
        encoded_bitarray = BitArray("")
        # make sure length is exact multiple of block size n
        assert (
            data_block.size % self.params.n == 0
        ), "Input to typical set encoder should be multiple of chunk length n"
        for chunk_idx in range(0, data_block.size // self.params.n):
            chunk_to_encode = tuple(
                data_block.data_list[chunk_idx * self.params.n : (chunk_idx + 1) * self.params.n]
            )
            if chunk_to_encode in self.index_typical:
                # typical case
                encoded_bitarray += "0"
                if self.index_bitlen_typical > 0:
                    encoded_bitarray += uint_to_bitarray(
                        self.index_typical[chunk_to_encode], self.index_bitlen_typical
                    )
            else:
                # non-typical case
                encoded_bitarray += "1"
                if self.index_bitlen_overall > 0:
                    encoded_bitarray += uint_to_bitarray(
                        self.index_overall[chunk_to_encode], self.index_bitlen_overall
                    )
        return encoded_bitarray


class TypicalSetDecoder(DataDecoder):
    """Typical set decoder. See module documentation for more details."""

    def __init__(self, params: TypicalSetCoderParams) -> None:
        super().__init__()
        self.params = params
        self.index_typical, self.index_overall = generate_typical_set_coder_lookup_tables(
            self.params
        )
        # compute lengths of the encoding of the index for typical and non-typical cases
        if len(self.index_typical) == 0:
            self.index_bitlen_typical = None  # never going to encounter
        else:
            self.index_bitlen_typical = int(np.ceil(np.log2(len(self.index_typical))))
        self.index_bitlen_overall = int(np.ceil(np.log2(len(self.index_overall))))

        # compute inverse index for decoding
        self.inverse_index_typical = {self.index_typical[s]: s for s in self.index_typical}
        self.inverse_index_overall = {self.index_overall[s]: s for s in self.index_overall}

    def decode_block(self, encoded_bitarray: BitArray):
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(encoded_bitarray):
            if encoded_bitarray[num_bits_consumed] == 0:
                # typical case
                num_bits_consumed += 1
                if self.index_bitlen_typical > 0:
                    index_typical = bitarray_to_uint(
                        encoded_bitarray[
                            num_bits_consumed : num_bits_consumed + self.index_bitlen_typical
                        ]
                    )
                else:
                    index_typical = 0  # if only one sequence
                num_bits_consumed += self.index_bitlen_typical
                data_list += self.inverse_index_typical[index_typical]
            else:
                # non-typical case
                num_bits_consumed += 1
                if self.index_bitlen_overall > 0:
                    index_overall = bitarray_to_uint(
                        encoded_bitarray[
                            num_bits_consumed : num_bits_consumed + self.index_bitlen_overall
                        ]
                    )
                else:
                    self.index_bitlen_overall = 0  # if only one sequence
                num_bits_consumed += self.index_bitlen_overall
                data_list += self.inverse_index_overall[index_overall]

        return DataBlock(data_list), num_bits_consumed


def test_is_typical():
    """
    Test the is_typical function for a given probability distribution
    and various inputs
    """
    prob_dist = ProbabilityDist({"A": 0.6, "B": 0.3, "C": 0.1})
    eps = 0.01
    n = 30

    # all A's is not typical
    chunk = ["A"] * n
    assert is_typical(chunk, prob_dist, eps) == False

    # 70% A's, 30% C's is not typical
    num_As = int(0.7 * n)
    num_Cs = n - num_As
    chunk = ["A"] * num_As + ["C"] * num_Cs
    assert is_typical(chunk, prob_dist, eps) == False
    # shuffling doesn't make a difference
    random.shuffle(chunk)
    assert is_typical(chunk, prob_dist, eps) == False
    random.shuffle(chunk)
    assert is_typical(chunk, prob_dist, eps) == False

    # 60% A's, 30% B's, 10% C's is typical
    num_As = int(0.6 * n)
    num_Bs = int(0.3 * n)
    num_Cs = n - num_As - num_Bs
    chunk = ["A"] * num_As + ["B"] * num_Bs + ["C"] * num_Cs
    assert is_typical(chunk, prob_dist, eps) == True
    # shuffling doesn't make a difference
    random.shuffle(chunk)
    assert is_typical(chunk, prob_dist, eps) == True
    random.shuffle(chunk)
    assert is_typical(chunk, prob_dist, eps) == True


def test_typical_set_coder_roundtrip():
    """
    Test roundtrip for typical set coder with various parameters
    """

    for prob_dist in [
        ProbabilityDist({"A": 0.6, "B": 0.3, "C": 0.1}),
        ProbabilityDist({"A": 0.6, "B": 0.4}),
        ProbabilityDist({"A": 0.99, "B": 0.01}),
        ProbabilityDist({"A": 1}),
    ]:
        for eps in [0.0, 0.01, 0.1, 0.2, 2.0]:
            for n in [1, 3, 6, 10]:
                data_block_size = 1000 * n
                data_block = get_random_data_block(prob_dist, data_block_size, seed=0)

                params = TypicalSetCoderParams(n, eps, prob_dist)

                # create encoder decoder
                encoder = TypicalSetEncoder(params)
                decoder = TypicalSetDecoder(params)

                # perform compression
                is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
                assert is_lossless, "Lossless compression failed"
