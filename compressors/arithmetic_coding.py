import numpy as np
from typing import Tuple
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies
from utils.test_utils import get_random_data_block, try_lossless_compression


class ArithmeticEncoder(DataEncoder):
    def __init__(self, precision, freqs):
        super().__init__()
        self.freqs = freqs

        # params
        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        self.FULL = 1 << (precision)
        self.HALF = 1 << (precision - 1)
        self.QTR = 1 << (precision - 2)

    def shrink_range(self, s, low, high) -> Tuple[int, int]:
        # rng, c, d
        rng = high - low
        c = self.freqs.cumulative_freq_dict[s]
        d = c + self.freqs.frequency(s)

        # perform shrinking
        high = low + (rng * d) // self.freqs.total_freq
        low = low + (rng * c) // self.freqs.total_freq
        return (low, high)

    def encode_block(self, data_block: DataBlock):
        # initialize the low and high states
        low = 0
        high = self.FULL

        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning
        encoded_bitarray = uint_to_bitarray(data_block.size, self.DATA_BLOCK_SIZE_BITS)

        # initialize counter for mid-range re-adjustments
        num_mid_range_readjust = 0

        # start the encoding
        for s in data_block.data_list:

            # shrink range
            low, high = self.shrink_range(s, low, high)

            while (high < self.HALF) or (low > self.HALF):
                if high < self.HALF:
                    # output 1's corresponding to prior mid-range readjustments
                    encoded_bitarray.extend("0" + "1" * num_mid_range_readjust)

                    # re-adjust range, and reset params
                    low = low << 1
                    high = high << 1
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

                elif low > self.HALF:
                    # output 0's corresponding to prior mid-range readjustments
                    encoded_bitarray.extend("1" + "0" * num_mid_range_readjust)

                    # re-adjust range, and reset params
                    low = (low - self.HALF) << 1
                    high = (high - self.HALF) << 1
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

            while (low > self.QTR) and (high < 3 * self.QTR):
                # increment the mid-range adjustment counter
                num_mid_range_readjust += 1
                low = (low - self.QTR) << 1
                high = (high - self.QTR) << 1

        # final flush
        num_mid_range_readjust += 1  # this increment is mainly to output either 01, 10
        if low <= self.QTR:
            # output 0's corresponding to prior mid-range readjustments
            encoded_bitarray.extend("0" + num_mid_range_readjust * "1")
        else:
            # output 1's corresponding to prior mid-range readjustments
            encoded_bitarray.extend("1" + num_mid_range_readjust * "0")

        return encoded_bitarray


class ArithmeticDecoder(DataDecoder):
    def __init__(self, precision, freqs):
        super().__init__()
        self.freqs = freqs

        # params
        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        self.FULL = 1 << (precision)
        self.HALF = 1 << (precision - 1)
        self.QTR = 1 << (precision - 2)

    def shrink_range(self, s, low, high) -> Tuple[int, int]:
        # rng, c, d
        rng = high - low
        c = self.freqs.cumulative_freq_dict[s]
        d = c + self.freqs.frequency(s)

        # perform shrinking
        high = low + (rng * d) // self.freqs.total_freq
        low = low + (rng * c) // self.freqs.total_freq
        return (low, high)

    def decode_symbol(self, low, high, state):
        rng = high - low
        search_list = (
            low
            + (np.array(list(self.freqs.cumulative_freq_dict.values())) * rng)
            // self.freqs.total_freq
        )
        start_bin = np.searchsorted(search_list, state, side="right") - 1
        s = self.freqs.alphabet[start_bin]
        return s

    def decode_block(self, encoded_bitarray: BitArray):

        data_block_size_bitarray = encoded_bitarray[: self.DATA_BLOCK_SIZE_BITS]
        encoded_bitarray = encoded_bitarray[self.DATA_BLOCK_SIZE_BITS :]

        # get data size
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)

        arith_bitarray_size = len(encoded_bitarray)

        # initialize return variables
        decoded_data_list = []
        num_bits_consumed = 0
        low = 0
        high = self.FULL
        state = 0

        # initialize the state
        while (num_bits_consumed < self.PRECISION) and (num_bits_consumed < arith_bitarray_size):
            bit = encoded_bitarray[num_bits_consumed]
            if bit:
                state += 1 << (self.PRECISION - num_bits_consumed - 1)
            num_bits_consumed += 1

        # main decoding loop
        while True:
            # decode the next symbol
            s = self.decode_symbol(low, high, state)
            low, high = self.shrink_range(s, low, high)
            decoded_data_list.append(s)

            # break when we have decoded all the symbols in the data block
            if len(decoded_data_list) == input_data_block_size:
                break

            while (high < self.HALF) or (low > self.HALF):
                if high < self.HALF:
                    # re-adjust range, and reset params
                    low = low << 1
                    high = high << 1
                    state = state << 1

                elif low > self.HALF:
                    # re-adjust range, and reset params
                    low = (low - self.HALF) << 1
                    high = (high - self.HALF) << 1
                    state = (state - self.HALF) << 1

                if num_bits_consumed < arith_bitarray_size:
                    bit = encoded_bitarray[num_bits_consumed]
                    state += bit
                    num_bits_consumed += 1

            while (low > self.QTR) and (high < 3 * self.QTR):
                # increment the mid-range adjustment counter
                low = (low - self.QTR) << 1
                high = (high - self.QTR) << 1
                state = (state - self.QTR) << 1

                if num_bits_consumed < arith_bitarray_size:
                    bit = encoded_bitarray[num_bits_consumed]
                    state += bit
                    num_bits_consumed += 1

        num_bits_consumed += self.DATA_BLOCK_SIZE_BITS
        return DataBlock(decoded_data_list), num_bits_consumed


def test_arithmetic_coding():
    freq = Frequencies({"A": 6, "B": 4, "C": 34})
    prob_dist = freq.get_prob_dist()

    # generate random data
    data_block = get_random_data_block(prob_dist, 100, seed=0)

    # create encoder decoder
    encoder = ArithmeticEncoder(32, freq)
    decoder = ArithmeticDecoder(32, freq)

    is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless
