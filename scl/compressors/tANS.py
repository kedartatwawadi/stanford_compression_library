"""tANS v1 (table ANS) implementation 

NOTE: tANS v1 is in a  way cached rANS implementation. There are other variants of tANS possible
See the wiki link: https://github.com/kedartatwawadi/stanford_compression_library/wiki/Asymmetric-Numeral-Systems
for more details on the algorithm

## References
1. Original Asymmetric Numeral Systems paper:  https://arxiv.org/abs/0902.0271
2. https://github.com/kedartatwawadi/stanford_compression_library/wiki/Asymmetric-Numeral-Systems
More references in the wiki article
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any, List
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import (
    BitArray,
    get_bit_width,
    uint_to_bitarray,
    bitarray_to_uint,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies, get_avg_neg_log_prob
from scl.utils.test_utils import get_random_data_block, try_lossless_compression
from scl.utils.misc_utils import cache, is_power_of_two
import pprint
from scl.compressors.rANS import rANSParams, rANSEncoder, rANSDecoder


@dataclass
class tANSParams(rANSParams):
    """
    NOTE: params are same as rANSParams
    we just restrict some parameter values
    """

    def __post_init__(self):
        super().__post_init__()
        ## restrict some param values for tANS
        # to make the rANS cachable, the total_freq needs to be a power of 2
        assert is_power_of_two(
            self.M
        ), "Please normalize self.M parameter (sum of frequencies) to be a power of two"

        # NOTE: NUM_BITS_OUT != 1, probably doesn't make practical sense for tANS
        # the reason is that, for M = total_freq, then [L, H] = [M, 2^{NUM_OUT_BITS}*M - 1]
        # thus the table size increases exponentially in NUM_OUT_BITS, which is typically bad
        assert self.NUM_BITS_OUT == 1, "only NUM_OUT_BITS = 1 supported for now"

        # just a warning to limit the table sizes
        if self.RANGE_FACTOR > (1 << 16):
            print("WARNING: RANGE_FACTOR > 2^16 --> the lookup tables could be huge")


class tANSEncoder(DataEncoder):
    """tANS Encoder (cached rANS version)"""

    def __init__(self, tans_params: tANSParams):
        """init function

        Args:
            tans_params (tANSParams): global tANS parameters
        """
        self.params = tans_params

        # build lookup tables
        self.build_base_encode_step_table()
        self.build_shrink_num_out_bits_lookup_table()

        # NOTE: uncomment to print and visualize the lookup tables
        # self._print_lookup_tables()

    def shrink_state_num_out_bits_base(self, s):
        """
        rANS shrink function either outputs n or n+1 number of bits
        , depending upon the state s.
        This function determines what n is here. More details on the wiki
        """
        # calculate the power of 2 lying in [freq[s], 2freq[s] - 1]
        y = get_bit_width(self.params.max_shrunk_state[s])
        num_out_bits_base = self.params.NUM_STATE_BITS - y

        # calculate the threshold to output 1 more bit
        thresh_state = (self.params.max_shrunk_state[s] + 1) << num_out_bits_base
        return num_out_bits_base, thresh_state

    def build_base_encode_step_table(self):
        """
        cached version of rans_encoder.rans_base_encode_step function
        """
        rans_encoder = rANSEncoder(self.params)
        self.base_encode_step_table = {}  # M rows, each storing x_next in [L,H]
        for s in self.params.freqs.alphabet:
            _min, _max = self.params.min_shrunk_state[s], self.params.max_shrunk_state[s]
            for x_shrunk in range(_min, _max + 1):
                self.base_encode_step_table[(s, x_shrunk)] = rans_encoder.rans_base_encode_step(
                    s, x_shrunk
                )

    def build_shrink_num_out_bits_lookup_table(self):
        """
        caching the shrink_state_num_out_bits_base function
        """
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
        """Encodes the next symbol, returns some bits and the updated state

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
        """
        main encode function. Does the following:
        1. sets state to INITIAL STATE
        2. recursively calls `state, symbol_bitarray = self.encode_symbol(s, state)` to update the state and
        output a few bits
        3. finally add the size of the input data using self.params.DATA_BLOCK_SIZE_BITS

        FIXME: this function is a duplicate or rANSEncoder, but duplicating it for clarity
        (we should remove once we know how to logically combine different rANS, tANS variants)
        """
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
        # NOTE: tANS decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = (
            uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS) + encoded_bitarray
        )

        return encoded_bitarray


class tANSDecoder(DataDecoder):
    """
    the table ANS decoder (implementing the cached rANS variant)
    """

    def __init__(self, tans_params: tANSParams):
        self.params = tans_params

        ## build lookup tables
        self.build_rans_base_decode_table()
        self.build_expand_state_num_bits_table()

    def build_rans_base_decode_table(self):
        """
        cache the rans_decoder.rans_base_decode_step function
        """
        rans_decoder = rANSDecoder(self.params)
        self.base_decode_step_table = {}  # stores s, state_shrunk
        for state in range(self.params.L, self.params.H + 1):
            self.base_decode_step_table[state] = rans_decoder.rans_base_decode_step(state)

    def build_expand_state_num_bits_table(self):
        """
        cache the expand state function
        """
        self.expand_state_num_bits_table = {}
        for s in self.params.freqs.alphabet:
            _min, _max = self.params.min_shrunk_state[s], self.params.max_shrunk_state[s]
            for x_shrunk in range(_min, _max + 1):
                num_bits = self.params.NUM_STATE_BITS - get_bit_width(x_shrunk)
                self.expand_state_num_bits_table[x_shrunk] = num_bits

    def _print_lookup_tables(self):
        """
        function useful to visualize the tANS tables + debugging
        """
        print("-" * 20)
        print("base decode step table")
        pprint.pprint(self.base_decode_step_table)
        print("-" * 20)
        print("Expand state num_bits table")
        pprint.pprint(self.expand_state_num_bits_table)

    def decode_symbol(self, state: int, encoded_bitarray: BitArray):
        # base rANS decoding step
        s, state_shrunk = self.base_decode_step_table[state]

        # remap the state into the acceptable range
        num_bits = self.expand_state_num_bits_table[state_shrunk]
        state_remainder = 0
        if num_bits:
            state_remainder = bitarray_to_uint(encoded_bitarray[:num_bits])
        state = (state_shrunk << num_bits) + state_remainder

        return s, state, num_bits

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


def test_generated_lookup_tables():
    # For a specific example to check if the lookup tables are as expected
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = tANSParams(freq, DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    ####################################################################
    # Encoder tests

    ## check if the encoder lookup tables are as expected
    # define expected tables
    expected_base_encode_step_table = {
        ("A", 3): 8,
        ("A", 4): 9,
        ("A", 5): 10,
        ("B", 3): 11,
        ("B", 4): 12,
        ("B", 5): 13,
        ("C", 2): 14,
        ("C", 3): 15,
    }
    expected_shrink_state_num_out_bits_base_table = {"A": 1, "B": 1, "C": 2}
    expected_shrink_state_thresh_table = {"A": 12, "B": 12, "C": 16}

    # check if the encoder lookup tables are the same as expected
    encoder = tANSEncoder(params)
    assert expected_base_encode_step_table == encoder.base_encode_step_table
    assert (
        expected_shrink_state_num_out_bits_base_table
        == encoder.shrink_state_num_out_bits_base_table
    )
    assert expected_shrink_state_thresh_table == encoder.shrink_state_thresh_table

    ####################################################################
    # Decoder tests

    ## check if the encoder lookup tables are as expected
    expected_base_decode_step_table = {
        8: ("A", 3),
        9: ("A", 4),
        10: ("A", 5),
        11: ("B", 3),
        12: ("B", 4),
        13: ("B", 5),
        14: ("C", 2),
        15: ("C", 3),
    }
    expected_expand_state_num_bits = {2: 2, 3: 2, 4: 1, 5: 1}

    # define expected tables
    decoder = tANSDecoder(params)
    assert expected_base_decode_step_table == decoder.base_decode_step_table
    assert expected_expand_state_num_bits == decoder.expand_state_num_bits_table


def test_check_encoded_bitarray():
    # test a specific example to check if the bitstream is as expected
    freq = Frequencies({"A": 3, "B": 3, "C": 2})
    data = DataBlock(["A", "C", "B"])
    params = tANSParams(freq, DATA_BLOCK_SIZE_BITS=5, NUM_BITS_OUT=1, RANGE_FACTOR=1)

    # Lets start by printing out the lookup tables:
    encoder = tANSEncoder(params)
    # print("*")
    # encoder._print_lookup_tables()

    ## Lookup tables:
    # --------------------
    # base encode step table
    # {('A', 3): 8,
    # ('A', 4): 9,
    # ('A', 5): 10,
    # ('B', 3): 11,
    # ('B', 4): 12,
    # ('B', 5): 13,
    # ('C', 2): 14,
    # ('C', 3): 15}
    # --------------------
    # num out bits table
    # {'A': 1, 'B': 1, 'C': 2}
    # --------------------
    # shrink state thresh
    # {'A': 12, 'B': 12, 'C': 16}

    # NOTE: the encoded_bitstream looks like = [<data_size_bits>, <final_state_bits>,<s0_bits>, <s1_bits>, ..., <s3_bits>]
    ## Lets manually encode to verify if the expected bitstream matches,

    expected_encoded_bitarray = BitArray("")

    # lets start with defining the initial_state
    x = 8
    assert params.INITIAL_STATE == 8

    ## encode symbol 1 = A
    # step-1: x < 12, so num_out_bits = 1
    x = 4  # x = x >> 1
    expected_encoded_bitarray = BitArray("0") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 9  # Looking at the base encode lookup table -> ('A', 4): 9

    ## encode symbol 2 = C
    # step-1: x < 16, num_out_bits = 2
    x = 2  # x = x >> 2
    expected_encoded_bitarray = BitArray("01") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 14  # ('C', 2): 14,

    ## encode symbol 3 = B
    # step-1: 14 >= shrink_state_thresh['B'] = 12, so
    # num_out_bits = 1 + 1 = 2
    x = 3  # x = x >> 2
    expected_encoded_bitarray = BitArray("10") + expected_encoded_bitarray

    # step-2: rANS base encoding step
    x = 11  # ('B', 3): 11

    ## prepend the final state to the bitarray
    num_state_bits = 4  # log2(15)
    assert params.NUM_STATE_BITS == num_state_bits
    expected_encoded_bitarray = BitArray("1011") + expected_encoded_bitarray

    # append number of symbols = 3 using params.DATA_BLOCK_SIZE_BITS
    expected_encoded_bitarray = BitArray("00011") + expected_encoded_bitarray

    ################################

    ## Now lets encode using the encode_block and see it the result matches
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
        tANSParams(freqs_list[1], RANGE_FACTOR=1 << 4),
        tANSParams(freqs_list[2]),
    ]

    # generate random data and test if coding is lossless
    DATA_SIZE = 10000
    SEED = 0
    for freq, tans_params in zip(freqs_list, params_list):
        # generate random data
        prob_dist = freq.get_prob_dist()
        data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)

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
