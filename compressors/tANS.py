"""tANS v1 (table ANS) implementation 

NOTE: tANS v1 is ina  way cached rANS implementation. There are other variants of tANS possible

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
from compressors.rANS import rANSParams, rANSEncoder


@dataclass
class tANSParams(rANSParams):
    # NOTE: params are same as rANSParams
    # we just restrict some parameter values

    def __post_init__(self):
        super().__post_init__()
        ## restrict some param values for tANS
        # to make the rANS cachable, the total_freq needs to be a power of 2
        assert is_power_of_two(
            self.freqs.total_freq
        ), "Please normalize self.freqs.total_freq to be a power of two"

        # NOTE: NUM_BITS_OUT != 1, probably doesn't make practical sense for tANS?
        assert self.NUM_BITS_OUT == 1, "only NUM_OUT_BITS = 1 supported for now"

        # just a warning to limit the table sizes
        if self.RANGE_FACTOR > (1 << 16):
            print("WARNING: RANGE_FACTOR > 2^16 --> the lookup tables could be huge")


class tANSEncoder(DataEncoder):
    """tANS Encoder (cached rANS version)"""

    def __init__(self, tans_params: tANSParams):
        """init function

        Args:
            tans_params (tANSParams): global tANS hyperparameters
        """
        self.params = tans_params

        # build lookup tables
        self.build_base_encode_step_table()
        self.build_shrink_num_out_bits_lookup_table()

        # NOTE: uncomment to print and visualize the lookup tables
        # self._print_lookup_tables()

    def shrink_state_num_out_bits_base(self, s):
        # calculate the power of 2 lying in [freq[s], 2freq[s] - 1]
        y = get_bit_width(self.params.max_shrunk_state[s])
        state_bits = get_bit_width(self.params.RANGE_FACTOR * self.params.freqs.total_freq)
        num_out_bits_base = state_bits - y

        # calculate the threshold to output 1 more bit
        thresh_state = (self.params.max_shrunk_state[s] + 1) << num_out_bits_base
        return num_out_bits_base, thresh_state

    def build_base_encode_step_table(self):
        rans_encoder = rANSEncoder(self.params)
        self.base_encode_step_table = {}  # M rows, each storing x_next in [L,H]
        for s in self.params.freqs.alphabet:
            _min, _max = self.params.min_shrunk_state[s], self.params.max_shrunk_state[s]
            for x_shrunk in range(_min, _max + 1):
                self.base_encode_step_table[(s, x_shrunk)] = rans_encoder.rans_base_encode_step(
                    s, x_shrunk
                )

    def build_shrink_num_out_bits_lookup_table(self):
        self.shrink_state_num_out_bits_base_table = {}
        self.shrink_state_thresh_table = {}
        for s in self.params.freqs.alphabet:
            num_bits, thresh = self.shrink_state_num_out_bits_base(s)
            self.shrink_state_num_out_bits_base_table[s] = num_bits
            self.shrink_state_thresh_table[s] = thresh

    def _print_lookup_tables(self):
        """
        function useful to visualize the tANS tables + debugging
        """
        print("-" * 20)
        print("base encode step table")
        pprint.pprint(self.base_encode_step_table)
        print("-" * 20)
        print("num out bits table")
        pprint.pprint(self.shrink_state_num_out_bits_base_table)
        print("-" * 20)
        print("shrink state thresh")
        pprint.pprint(self.shrink_state_thresh_table)

    def encode_symbol(self, s, state: int) -> Tuple[int, BitArray]:
        """Encodes the next symbol, returns some bits and  the updated state

        In the tANS encode_symbol, note that all we are doing is accessing lookup tables.
        The lookup tables are already defined during the init

        Args:
            s (Any): next symbol to be encoded
            state (int): the rANS state

        Returns:
            state (int), symbol_bitarray (BitArray):
        """
        # output bits to the stream so that the state is in the acceptable range
        # [L, H] *after*the `rans_base_encode_step`
        symbol_bitarray = BitArray("")

        # shrink state x before calling base encode
        num_out_bits = self.shrink_state_num_out_bits_base_table[s]
        if state >= self.shrink_state_thresh_table[s]:
            num_out_bits += 1

        out_bits = uint_to_bitarray(state)[-num_out_bits:] if num_out_bits else BitArray("")
        state = state >> num_out_bits

        # NOTE: we are prepending bits for pedagogy. In practice, it might be faster to assign a larger memory chunk and then fill it from the back
        # see: https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/main.cpp#L176 for example
        symbol_bitarray = out_bits + symbol_bitarray

        # core encoding step
        state = self.base_encode_step_table[(s, state)]
        return state, symbol_bitarray

    def encode_block(self, data_block: DataBlock):
        # initialize the output
        encoded_bitarray = BitArray("")

        # initialize the state
        state = self.params.INITIAL_STATE

        # update the state
        for s in data_block.data_list:
            state, symbol_bitarray = self.encode_symbol(s, state)
            encoded_bitarray = symbol_bitarray + encoded_bitarray

        # Finally, pre-pend binary representation of the final state
        encoded_bitarray = uint_to_bitarray(state, self.params.NUM_STATE_BITS) + encoded_bitarray

        # add the data_block size at the beginning
        # NOTE: rANS decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = (
            uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS) + encoded_bitarray
        )

        return encoded_bitarray


class tANSDecoder(DataDecoder):
    def __init__(self, tans_params: tANSParams):
        self.params = tans_params

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
        block_id = state // self.params.freqs.total_freq
        slot = state % self.params.freqs.total_freq

        # decode symbol
        cum_prob_list = list(self.params.freqs.cumulative_freq_dict.values())
        symbol_ind = self.find_bin(cum_prob_list, slot)
        s = self.params.freqs.alphabet[symbol_ind]

        # retrieve prev state
        prev_state = (
            block_id * self.params.freqs.frequency(s)
            + slot
            - self.params.freqs.cumulative_freq_dict[s]
        )
        return s, prev_state

    def expand_state(self, state: int, encoded_bitarray: BitArray) -> Tuple[int, int]:
        # remap the state into the acceptable range
        num_bits = 0
        while state < self.params.L:
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
        state = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + self.params.NUM_STATE_BITS]
        )
        num_bits_consumed += self.params.NUM_STATE_BITS

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
        assert state == self.params.INITIAL_STATE

        return DataBlock(decoded_data_list), num_bits_consumed


######################################## TESTS ##########################################


def test_check_encoded_bitarray():
    # test a specific example to check if the bitstream is as expected
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = tANSParams(freq, DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    # NOTE: the encoded_bitstream looks like = [<data_size_bits>, <final_state_bits>,<s0_bits>, <s1_bits>, ..., <s3_bits>]
    ## Lets manually encode to find intermediate state etc:
    M = 8  # freq.total_freq
    L = 8  # = Mt
    H = 15  # = 2Mt-1

    expected_encoded_bitarray = BitArray("")

    # lets start with defining the initial_state
    x = 8
    assert params.INITIAL_STATE == 8

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
    assert params.NUM_STATE_BITS == num_state_bits
    expected_encoded_bitarray = BitArray("1011") + expected_encoded_bitarray

    # append number of symbols = 3 using params.DATA_BLOCK_SIZE_BITS
    expected_encoded_bitarray = BitArray("00011") + expected_encoded_bitarray

    ################################

    ## Now lets encode using the encode_block and see it the result matches
    encoder = tANSEncoder(params)
    encoded_bitarray = encoder.encode_block(data)

    assert expected_encoded_bitarray == encoded_bitarray


def test_tANS_coding():
    ## List different distributions, rANS params to test
    # trying out some random frequencies
    freqs_list = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 3}),
        Frequencies({"A": 3, "B": 4, "C": 9}),
    ]
    params_list = [
        tANSParams(freqs_list[0], RANGE_FACTOR=1),
        tANSParams(freqs_list[1], RANGE_FACTOR=1 << 8),
        tANSParams(freqs_list[2]),
    ]

    # generate random data and test if coding is lossless
    DATA_SIZE = 10000
    SEED = 0
    for freq, tans_params in zip(freqs_list, params_list):
        # generate random data
        prob_dist = freq.get_prob_dist()
        data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)
        avg_log_prob = get_mean_log_prob(prob_dist, data_block)

        # create encoder decoder
        encoder = tANSEncoder(tans_params)
        decoder = tANSDecoder(tans_params)

        # test lossless coding
        is_lossless, encode_len, _ = try_lossless_compression(
            data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
        )
        assert is_lossless
        # avg codelen ignoring the bits used to signal num data elements
        avg_codelen = encode_len / data_block.size
        print(f"tANS coding: avg_log_prob={avg_log_prob:.3f}, tANS codelen: {avg_codelen:.3f}")
