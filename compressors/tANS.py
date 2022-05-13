"""Streaming rANS (range Asymmetric Numeral Systems) implementation

M = 2^r
L = M
H = 2L - 1

def rans_base_encode_step(x_shrunk,s):
    ...
    return x_next

def shrink_state_num_out_bits_base(s):
    # calculate the power of 2 lying in [freq[s], 2freq[s] - 1]
    y = ceil(log2(freq[s]))
    return y

### build the tables ###
# NOTE: this is a one time thing which can be done at the beginning of encoding
# or at compile time

base_encode_step_table = {} #M rows, each storing x_next in [L,H]
for s in Alphabet:
    for x_shrunk in Interval[freq[s], 2*freq[s] - 1]:
        base_encode_step_table[x_shrunk, s] = rans_base_encode_step(x_shrunk,s)

shrink_state_num_out_bits_base = {} #stores the exponent y values as described above
shrink_state_thresh = {} # stores the thresh values
for s in Alphabet:
    shrink_state_num_out_bits_base[s] = shrink_state_num_out_bits_base(s)
    shrink_state_thresh[s] = 2**shrink_state_num_out_bits_base[s]

### the cached encode_step ########
def encode_step_cached(x,s):
    # shrink state x before calling base encode
    num_out_bits = shrink_state_num_out_bits_base[s]
    if x < shrink_state_thresh[s]:
        num_out_bits += 1
    x_shrunk = x >> num_out_bits
    out_bits = to_binary(x)[-num_out_bits:]

    # perform the base encoding step
    x_next = base_encode_step_table[x_shrunk,s]
   
    return x_next, out_bits

## References
1. Original Asymmetric Numeral Systems paper:  https://arxiv.org/abs/0902.0271
2. https://github.com/kedartatwawadi/stanford_compression_library/wiki/Asymmetric-Numeral-Systems
More references in the wiki article
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any, List
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, get_bit_width, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies, get_mean_log_prob
from utils.test_utils import get_random_data_block, try_lossless_compression
from utils.misc_utils import cache, is_power_of_two
import pprint


@dataclass
class rANSParams:
    """base parameters for the rANS encoder/decoder.
    More details in the overview
    """

    # num bits used to represent the data_block size
    DATA_BLOCK_SIZE_BITS: int = 32

    # the encoder can output NUM_BITS_OUT at a time when it performs the state shrinking operation
    NUM_BITS_OUT: int = 1  # number of bits

    # rANS state is limited to the range [RANGE_FACTOR*total_freq, (2**NUM_BITS_OUT)*RANGE_FACTOR*total_freq - 1)]
    # RANGE_FACTOR is a base parameter controlling this range
    RANGE_FACTOR_BITS: int = 16
    RANGE_FACTOR: int = 1 << RANGE_FACTOR_BITS

    def initial_state(self, freqs: Frequencies) -> int:
        """the initial state from which rANS encoding begins

        NOTE: the choice of  this state is somewhat arbitrary, the only condition being, it should lie in the acceptable range
        [L, H]
        """
        return self.RANGE_FACTOR * freqs.total_freq

    def num_state_bits(self, total_freqs: int) -> int:
        """returns the number of bits necessary to represent the rANS state
        NOTE:  in rANS, the state is always limited to be in the range [L, H], where:
        L = RANGE_FACTOR*total_freq
        H = (2**NUM_BITS_OUT)*RANGE_FACTOR*total_freq - 1)

        Args:
            total_freqs (int): sum of all the frequencies of the alphabet

        Returns:
            int: the number of bits required to represent the rANS state
        """
        max_state_size = self.RANGE_FACTOR * (1 << self.NUM_BITS_OUT) * total_freqs - 1
        return get_bit_width(max_state_size)


class rANSEncoder(DataEncoder):
    """rANS Encoder

    Detailed information in the overview
    """

    def __init__(self, freqs: Frequencies, rans_params: rANSParams):
        """init function

        Args:
            freqs (Frequencies): frequencies for which rANS encoder needs to be designed
            rans_params (rANSParams): global rANS hyperparameters
        """
        self.freqs = freqs
        assert is_power_of_two(
            self.freqs.total_freq
        ), "Please normalize self.freqs.total_freq to be a power of two"

        self.params = rans_params
        assert self.params.NUM_BITS_OUT == 1, "only NUM_OUT_BITS = 1 supported for now"

        # build lookup tables
        self.build_base_encode_step_table()
        self.build_shrink_num_out_bits_lookup_table()
        print("-" * 20)
        print("base encode step table")
        pprint.pprint(self.base_encode_step_table)
        print("-" * 20)
        print("num out bits table")
        pprint.pprint(self.shrink_state_num_out_bits_base_table)
        print("-" * 20)
        print("shrink state thresh")
        pprint.pprint(self.shrink_state_thresh_table)

        breakpoint()

    def rans_base_encode_step(self, s, state: int):
        """base rANS encode step

        updates the state based on the input symbols s, and returns the updated state
        """
        f = self.freqs.frequency(s)
        block_id = state // f
        slot = self.freqs.cumulative_freq_dict[s] + (state % f)
        next_state = block_id * self.freqs.total_freq + slot
        return next_state

    def min_shrunk_state_val(self, symbol):
        """
        max value the state can be before calling rans_base_encode_step function
        """
        f = self.freqs.frequency(symbol)
        return self.params.RANGE_FACTOR * f

    def max_shrunk_state_val(self, symbol):
        """
        max value the state can be before calling rans_base_encode_step function
        """
        f = self.freqs.frequency(symbol)
        return self.params.RANGE_FACTOR * f * (1 << self.params.NUM_BITS_OUT) - 1

    def shrink_state_num_out_bits_base(self, s):
        # calculate the power of 2 lying in [freq[s], 2freq[s] - 1]
        y = get_bit_width(self.max_shrunk_state_val(s))
        state_bits = get_bit_width(self.params.RANGE_FACTOR * self.freqs.total_freq)
        num_out_bits_base = state_bits - y
        thresh_state = (self.max_shrunk_state_val(s) + 1) << num_out_bits_base
        return num_out_bits_base, thresh_state

    def build_base_encode_step_table(self):
        self.base_encode_step_table = {}  # M rows, each storing x_next in [L,H]
        for s in self.freqs.alphabet:
            _min = self.min_shrunk_state_val(s)
            _max = self.max_shrunk_state_val(s)
            for x_shrunk in range(_min, _max + 1):
                self.base_encode_step_table[(s, x_shrunk)] = self.rans_base_encode_step(s, x_shrunk)

    def build_shrink_num_out_bits_lookup_table(self):
        self.shrink_state_num_out_bits_base_table = (
            {}
        )  # stores the exponent y values as described above
        self.shrink_state_thresh_table = {}
        for s in self.freqs.alphabet:
            num_bits, thresh = self.shrink_state_num_out_bits_base(s)
            self.shrink_state_num_out_bits_base_table[s] = num_bits
            self.shrink_state_thresh_table[s] = thresh

    def encode_symbol(self, s, state: int) -> Tuple[int, BitArray]:
        """Encodes the next symbol, returns some bits and  the updated state

        Args:
            s (Any): next symbol to be encoded
            state (int): the rANS state

        Returns:
            state (int), symbol_bitarray (BitArray):
        """
        # output bits to the stream so that the state is in the acceptable range
        # [L, H] *after*the `rans_base_encode_step`
        print(state)
        symbol_bitarray = BitArray("")

        # shrink state x before calling base encode
        num_out_bits = self.shrink_state_num_out_bits_base_table[s]
        if state >= self.shrink_state_thresh_table[s]:
            num_out_bits += 1
        out_bits = uint_to_bitarray(state)[-num_out_bits:]
        state = state >> num_out_bits

        # NOTE: we are prepending bits for pedagogy. In practice, it might be faster to assign a larger memory chunk and then fill it from the back
        # see: https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/main.cpp#L176 for example
        symbol_bitarray = out_bits + symbol_bitarray

        # core encoding step
        state = self.base_encode_step_table[(s, state)]
        print(s, state, symbol_bitarray)
        return state, symbol_bitarray

    def encode_block(self, data_block: DataBlock):
        # initialize the output
        encoded_bitarray = BitArray("")

        # initialize the state
        state = self.params.initial_state(self.freqs)

        # update the state
        for s in data_block.data_list:
            state, symbol_bitarray = self.encode_symbol(s, state)
            encoded_bitarray = symbol_bitarray + encoded_bitarray

        # Finally, pre-pend binary representation of the final state
        num_state_bits = self.params.num_state_bits(self.freqs.total_freq)
        encoded_bitarray = uint_to_bitarray(state, num_state_bits) + encoded_bitarray

        # add the data_block size at the beginning
        # NOTE: rANS decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = (
            uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS) + encoded_bitarray
        )

        return encoded_bitarray


class rANSDecoder(DataDecoder):
    def __init__(self, freqs: Frequencies, rans_params: rANSParams):
        self.freqs = freqs
        self.params = rans_params

        # the range in which the state lies
        self.L = self.params.RANGE_FACTOR * self.freqs.total_freq
        self.H = self.L * (1 << self.params.NUM_BITS_OUT) - 1

    @staticmethod
    def find_bin(cumulative_freqs_list: List, slot: int) -> int:
        """Performs binary search over cumulative_freqs_list to locate which bin
        the slot lies.

        Args:
            cumulative_freqs_list (List): the sorted list of cumulative frequencies
                For example: freqs_list = [2,7,3], cumulative_freqs_list [0,2,9]
            slot (int): the value to search in the sorted list

        Returns:
            bin: the bin in which the slot lies
        """
        # NOTE: side="right" corresponds to search of type a[i-1] <= t < a[i]
        bin = np.searchsorted(cumulative_freqs_list, slot, side="right") - 1
        return int(bin)

    def rans_base_decode_step(self, state: int):
        block_id = state // self.freqs.total_freq
        slot = state % self.freqs.total_freq

        # decode symbol
        cum_prob_list = list(self.freqs.cumulative_freq_dict.values())
        symbol_ind = self.find_bin(cum_prob_list, slot)
        s = self.freqs.alphabet[symbol_ind]

        # retrieve prev state
        prev_state = block_id * self.freqs.frequency(s) + slot - self.freqs.cumulative_freq_dict[s]
        return s, prev_state

    def expand_state(self, state: int, encoded_bitarray: BitArray) -> Tuple[int, int]:
        # remap the state into the acceptable range
        num_bits = 0
        while state < self.L:
            state_remainder = bitarray_to_uint(
                encoded_bitarray[num_bits : num_bits + self.params.NUM_BITS_OUT]
            )
            num_bits += self.params.NUM_BITS_OUT
            state = (state << self.params.NUM_BITS_OUT) + state_remainder
        return state, num_bits

    def decode_symbol(self, state: int, encoded_bitarray: BitArray):
        # base rANS decoding step
        s, state = self.rans_base_decode_step(state)

        # remap the state into the acceptable range
        state, num_bits_used_by_expand_state = self.expand_state(state, encoded_bitarray)
        return s, state, num_bits_used_by_expand_state

    def decode_block(self, encoded_bitarray: BitArray):
        # get data block size
        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)
        num_bits_consumed = self.params.DATA_BLOCK_SIZE_BITS

        # get the final state
        num_state_bits = self.params.num_state_bits(self.freqs.total_freq)
        state = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + num_state_bits]
        )
        num_bits_consumed += num_state_bits

        # perform the decoding
        decoded_data_list = []
        for _ in range(input_data_block_size):
            s, state, num_symbol_bits = self.decode_symbol(
                state, encoded_bitarray[num_bits_consumed:]
            )

            # rANS decoder decodes symbols in the reverse direction,
            # so we add newly decoded symbol at the beginning
            decoded_data_list = [s] + decoded_data_list
            num_bits_consumed += num_symbol_bits

        # Finally, as a sanity check, ensure that the end state should be equal to the initial state
        assert state == self.params.initial_state(self.freqs)

        return DataBlock(decoded_data_list), num_bits_consumed


######################################## TESTS ##########################################


def _test_rANS_coding(freq, rans_params, data_size, seed):
    """Core testing function for rANS"""
    prob_dist = freq.get_prob_dist()

    # generate random data
    data_block = get_random_data_block(prob_dist, data_size, seed=seed)

    # get optimal codelen
    avg_log_prob = get_mean_log_prob(prob_dist, data_block)

    # create encoder decoder
    encoder = rANSEncoder(freq, rans_params)
    decoder = rANSDecoder(freq, rans_params)

    is_lossless, encode_len, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )

    # avg codelen ignoring the bits used to signal num data elements
    avg_codelen = encode_len / data_block.size
    print(f"rANS coding: Optical codelen={avg_log_prob:.3f}, rANS codelen: {avg_codelen:.3f}")
    assert is_lossless


def test_check_encoded_bitarray():
    # test a specific example to check if the bitstream is as expected
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = rANSParams(DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    # NOTE: the encoded_bitstream looks like = [<data_size_bits>, <final_state_bits>,<s0_bits>, <s1_bits>, ..., <s3_bits>]
    ## Lets manually encode to find intermediate state etc:
    M = 8  # freq.total_freq
    L = 8  # = Mt
    H = 15  # = 2Mt-1

    expected_encoded_bitarray = BitArray("")

    # lets start with defining the initial_state
    x = 8
    assert params.initial_state(freq) == 8

    ## encode symbol 1 = A
    # step-1: shrink state x to be in [3, 5]
    x = 4  # x = x//2
    expected_encoded_bitarray = BitArray("0") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 9  # x = (x//3)*8 + 0 + (x%3)

    ## encode symbol 2 = C
    # step-1: shrink state x to be in [2, 3]
    x = 2  # x = x//4
    expected_encoded_bitarray = BitArray("01") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 14  # x = (x//2)*8 + 6 + (x%2)

    ## encode symbol 3 = B
    # step-1: shrink state x to be in [3, 5]
    x = 3  # x = x//4
    expected_encoded_bitarray = BitArray("10") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 11  # x = (x//3)*8 + 3 + (x%3)

    ## prepnd the final state to the bitarray
    num_state_bits = 4  # log2(15)
    assert params.num_state_bits(freq.total_freq) == num_state_bits
    expected_encoded_bitarray = BitArray("1011") + expected_encoded_bitarray

    # append number of symbols = 3 using params.DATA_BLOCK_SIZE_BITS
    expected_encoded_bitarray = BitArray("00011") + expected_encoded_bitarray

    ################################

    ## Now lets encode using the encode_block and see it the result matches
    encoder = rANSEncoder(freq, params)
    encoded_bitarray = encoder.encode_block(data)

    assert expected_encoded_bitarray == encoded_bitarray


def test_rANS_coding():

    ## Test lossless coding
    DATA_SIZE = 10
    # trying out some random frequencies
    freqs = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        # Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        # Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        # Frequencies({"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5}),
    ]

    params = [
        rANSParams(),
        # rANSParams(),
        # rANSParams(NUM_BITS_OUT=8),
        # rANSParams(RANGE_FACTOR_BITS=12),
    ]
    # for freq, param in zip(freqs, params):
    #     _test_rANS_coding(freq, param, DATA_SIZE, seed=0)
