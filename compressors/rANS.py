import numpy as np
from typing import Tuple, Any
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies, get_mean_log_prob
from utils.test_utils import get_random_data_block, try_lossless_compression


class rANSEncoderInfinitePrecision(DataEncoder):
    def __init__(self, freqs: Frequencies):
        self.freqs = freqs
        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block


    def rans_base_encode_step(self, s, state):
        f = self.freqs.frequency(s)
        block_id = (state // f)
        slot = self.freqs.cumulative_freq_dict[s] + (state % f)
        next_state = block_id * self.freqs.total_freq + slot 
        return next_state

    def encode_block(self, data_block: DataBlock):
        
        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning
        # NOTE: rANS decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = uint_to_bitarray(data_block.size, self.DATA_BLOCK_SIZE_BITS)

        # initialize the state
        state = 0

        # update the state
        for s in data_block.data_list:
            state = self.rans_base_encode_step(s, state)
            print(state)
        # get binary representation of the state
        encoded_bitarray += uint_to_bitarray(state)
        return encoded_bitarray


class rANSDecoderInfinitePrecision(DataDecoder):
    def __init__(self, freqs: Frequencies):
        self.freqs = freqs
        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block

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
        prev_state = block_id*self.freqs.frequency(s) + slot - self.freqs.cumulative_freq_dict[s]
        return s, prev_state
    
    def decode_block(self, encoded_bitarray: BitArray):
        data_block_size_bitarray = encoded_bitarray[: self.DATA_BLOCK_SIZE_BITS]
        state_bitarray = encoded_bitarray[self.DATA_BLOCK_SIZE_BITS :]

        # get data size
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)

        # get the final state
        state = bitarray_to_uint(state_bitarray)
        print(state)


        # initialize return variables
        decoded_data_list = []
        num_bits_decoded = len(encoded_bitarray) #NOTE: rANS does not know when to "stop"

        for i in range(input_data_block_size):
            s, state = self.rans_base_decode_step(state)
            decoded_data_list.append(s)
            print(state)
        
        # the final state should be equal to the initial state
        assert state == 0
        
        decoded_data_list.reverse()

        return DataBlock(decoded_data_list), num_bits_decoded



def _test_rANS_coding(freq, data_size, seed):
    prob_dist = freq.get_prob_dist()

    # generate random data
    data_block = get_random_data_block(prob_dist, data_size, seed=0)

    # get optimal codelen
    avg_log_prob = get_mean_log_prob(prob_dist, data_block)

    # create encoder decoder
    data_size_bits = 32
    encoder = rANSEncoderInfinitePrecision(freq)
    decoder = rANSDecoderInfinitePrecision(freq)

    is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder)

    # avg codelen ignoring the bits used to signal num data elements
    avg_codelen = (encode_len - data_size_bits) / data_block.size
    print(
        f"rANS coding: Optical codelen={avg_log_prob:.3f}, AEC codelen(without header): {avg_codelen:.3f}"
    )

    # # check whether arithmetic coding results are close to optimal codelen
    # np.testing.assert_almost_equal(
    #     avg_codelen,
    #     avg_log_prob,
    #     decimal=2,
    #     err_msg="Arithmetic coding is not close to avg codelen",
    # )
    assert is_lossless


def test_rANS_coding():
    DATA_SIZE = 10

    # trying out some random frequencies
    freqs = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5}),
    ]

    for freq in freqs:
        _test_rANS_coding(freq, DATA_SIZE, seed=0)


