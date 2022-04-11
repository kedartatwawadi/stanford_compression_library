from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, get_bit_width, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies, get_mean_log_prob
from utils.test_utils import get_random_data_block, try_lossless_compression


@dataclass
class rANSParams:
    DATA_BLOCK_SIZE_BITS: int = 32
    RANGE_FACTOR_BITS: int = 16
    RANGE_FACTOR: int = 1 << RANGE_FACTOR_BITS
    NUM_BITS_OUT: int = 1

    def num_state_bits(self, total_freqs):
        """_summary_

        Args:
            total_freqs (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_state_size = self.RANGE_FACTOR * (1 << self.NUM_BITS_OUT) * total_freqs - 1
        return get_bit_width(max_state_size)


class rANSEncoder(DataEncoder):
    def __init__(self, freqs: Frequencies):
        self.freqs = freqs
        self.params = rANSParams()

    def rans_base_encode_step(self, s, state):
        f = self.freqs.frequency(s)
        block_id = state // f
        slot = self.freqs.cumulative_freq_dict[s] + (state % f)
        next_state = block_id * self.freqs.total_freq + slot
        return next_state

    def encode_block(self, data_block: DataBlock):

        # initialize the output
        encoded_bitarray = BitArray("")

        # initialize the state
        state = self.params.RANGE_FACTOR * self.freqs.total_freq

        # update the state
        for s in data_block.data_list:
            f = self.freqs.frequency(s)

            # output bits to the stream to bring the state in the range for the next encoding
            max_state_val = self.params.RANGE_FACTOR * f * (1 << self.params.NUM_BITS_OUT) - 1
            while state > max_state_val:
                out_bits = uint_to_bitarray(
                    state % (1 << self.params.NUM_BITS_OUT), bit_width=self.params.NUM_BITS_OUT
                )
                encoded_bitarray = out_bits + encoded_bitarray
                state = state >> self.params.NUM_BITS_OUT

            # core encoding step
            state = self.rans_base_encode_step(s, state)

        # pre-pend binary representation of the state
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
    def __init__(self, freqs: Frequencies):
        self.freqs = freqs
        self.params = rANSParams()

    @staticmethod
    def find_bin(cumulative_freqs_list, slot):
        bin = np.searchsorted(cumulative_freqs_list, slot, side="right") - 1
        return bin

    def rans_base_decode_step(self, state):
        block_id = state // self.freqs.total_freq
        slot = state % self.freqs.total_freq

        # decode symbol
        cum_prob_list = list(self.freqs.cumulative_freq_dict.values())
        symbol_ind = self.find_bin(cum_prob_list, slot)
        s = self.freqs.alphabet[symbol_ind]

        # retrieve prev state
        prev_state = block_id * self.freqs.frequency(s) + slot - self.freqs.cumulative_freq_dict[s]
        return s, prev_state

    def decode_block(self, encoded_bitarray: BitArray):
        # get data size
        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)
        num_bits_consumed = self.params.DATA_BLOCK_SIZE_BITS

        # get the final state
        num_state_bits = self.params.num_state_bits(self.freqs.total_freq)
        state = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + num_state_bits]
        )
        num_bits_consumed += num_state_bits

        # initialize return variables
        decoded_data_list = []

        # perform the decoding
        for it in range(input_data_block_size):
            s, state = self.rans_base_decode_step(state)
            decoded_data_list.append(s)

            # remap the state into the acceptable range
            f = self.freqs.frequency(s)
            while state < self.params.RANGE_FACTOR * self.freqs.total_freq:
                state_remainder = bitarray_to_uint(
                    encoded_bitarray[
                        num_bits_consumed : num_bits_consumed + self.params.NUM_BITS_OUT
                    ]
                )
                num_bits_consumed += self.params.NUM_BITS_OUT
                state = (state << self.params.NUM_BITS_OUT) + state_remainder

        # the final state should be equal to the initial state
        assert state == self.params.RANGE_FACTOR * self.freqs.total_freq
        decoded_data_list.reverse()

        return DataBlock(decoded_data_list), num_bits_consumed


def _test_rANS_coding(freq, data_size, seed):
    prob_dist = freq.get_prob_dist()

    # generate random data
    data_block = get_random_data_block(prob_dist, data_size, seed=seed)

    # get optimal codelen
    avg_log_prob = get_mean_log_prob(prob_dist, data_block)

    # create encoder decoder
    data_size_bits = 32
    encoder = rANSEncoder(freq)
    decoder = rANSDecoder(freq)

    is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder)

    # avg codelen ignoring the bits used to signal num data elements
    avg_codelen = (encode_len - data_size_bits) / data_block.size
    print(
        f"rANS coding: Optical codelen={avg_log_prob:.3f}, rANS codelen(without header): {avg_codelen:.3f}"
    )
    assert is_lossless


def test_rANS_coding():
    DATA_SIZE = 10000

    # trying out some random frequencies
    freqs = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5}),
    ]

    for freq in freqs:
        _test_rANS_coding(freq, DATA_SIZE, seed=0)
