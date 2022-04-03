import numpy as np
from typing import Tuple, Any
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies
from utils.test_utils import get_random_data_block, try_lossless_compression


class ArithmeticEncoder(DataEncoder):
    """Finite precision Arithmetic encoder

    The encoder are decoders are based on the following sources:
    - https://youtu.be/ouYV3rBtrTI: This series of videos on Arithmetic coding are a very gradual but a great
    way to understand them
    - Charles Bloom's blog: https://www.cbloom.com/algs/statisti.html#A5
    - There is of course the original paper: ADD LINK
    """

    def __init__(self, precision, freqs):
        super().__init__()
        self.freqs = freqs

        # params
        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        self.FULL = 1 << (precision)
        self.HALF = 1 << (precision - 1)
        self.QTR = 1 << (precision - 2)

    def shrink_range(self, s: Any, low: int, high: int) -> Tuple[int, int]:
        """shrinks the range (low, high) based on the symbol s

        Args:
            s (Any): symbol to encode

        Returns:
            Tuple[int, int]: (low, high) ranges returned after shrinking
        """
        # compute some intermediate variables: rng, c, d
        rng = high - low
        c = self.freqs.cumulative_freq_dict[s]
        d = c + self.freqs.frequency(s)

        # perform shrinking of low, high
        # NOTE: this is the basic Arithmetic coding step implemented using integers
        high = low + (rng * d) // self.freqs.total_freq
        low = low + (rng * c) // self.freqs.total_freq
        return (low, high)

    def encode_block(self, data_block: DataBlock):
        """Encode block function for arithmetic coding"""

        # initialize the low and high states
        low = 0
        high = self.FULL

        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning
        # NOTE: Arithmetic decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = uint_to_bitarray(data_block.size, self.DATA_BLOCK_SIZE_BITS)

        # initialize counter for mid-range re-adjustments
        num_mid_range_readjust = 0

        # start the encoding
        for s in data_block.data_list:

            # shrink range
            # i.e. the core Arithmetic encoding step
            low, high = self.shrink_range(s, low, high)

            # perform re-normalizing range
            # NOTE: the low, high values need to be re-normalized as else they will keep shrinking
            # and after a few iterations things will be infeasible.
            # The goal of re-normalizing is to not let the range (high - low) get smaller than self.QTR

            # CASE I, II -> simple cases where low, high are both in the same half
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

            # CASE III -> the more complex case where low, high straddle the midpoint
            while (low > self.QTR) and (high < 3 * self.QTR):
                # increment the mid-range adjustment counter
                num_mid_range_readjust += 1
                low = (low - self.QTR) << 1
                high = (high - self.QTR) << 1

        # Finally output a few bits to signal the final range + any remaining mid range readjustments
        num_mid_range_readjust += 1  # this increment is mainly to output either 01, 10
        if low <= self.QTR:
            # output 0's corresponding to prior mid-range readjustments
            encoded_bitarray.extend("0" + num_mid_range_readjust * "1")
        else:
            # output 1's corresponding to prior mid-range readjustments
            encoded_bitarray.extend("1" + num_mid_range_readjust * "0")

        return encoded_bitarray


class ArithmeticDecoder(DataDecoder):
    """Finite precision Arithmetic decoder

    The encoder are decoders are based on the following sources:
    - https://youtu.be/ouYV3rBtrTI: This series of videos on Arithmetic coding are a very gradual but a great
    way to understand them
    - Charles Bloom's blog: https://www.cbloom.com/algs/statisti.html#A5
    - There is of course the original paper: ADD LINK
    """

    def __init__(self, precision, freqs):
        super().__init__()
        self.freqs = freqs

        # params
        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        self.FULL = 1 << (precision)
        self.HALF = 1 << (precision - 1)
        self.QTR = 1 << (precision - 2)

    def shrink_range(self, s: Any, low: int, high: int) -> Tuple[int, int]:
        """shrinks the range (low, high) based on the symbol s

        Args:
            s (Any): symbol to encode

        Returns:
            Tuple[int, int]: (low, high) ranges returned after shrinking
        """
        # compute some intermediate variables: rng, c, d
        rng = high - low
        c = self.freqs.cumulative_freq_dict[s]
        d = c + self.freqs.frequency(s)

        # perform shrinking of low, high
        # NOTE: this is the basic Arithmetic coding step implemented using integers
        high = low + (rng * d) // self.freqs.total_freq
        low = low + (rng * c) // self.freqs.total_freq
        return (low, high)

    def decode_symbol(self, low: int, high: int, state: int):
        """Core Arithmetic decoding function

        We cut the [low, high) range bits proportional to the cumulative probability of each symbol
        the function locates the bin in which the state lies
        NOTE: This is exactly same as the decoding function of the theoretical arithmetic decoder,
        except implemented using integers

        Args:
            low (int): range low point
            high (int): range high point
            state (int): the arithmetic decoder state

        Returns:
            s : the decoded symbol
        """

        # FIXME: simplify this search.
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
    freq = Frequencies({"A": 1, "B": 1, "C": 2})
    prob_dist = freq.get_prob_dist()

    # generate random data
    data_block = get_random_data_block(prob_dist, 2000, seed=0)

    # create encoder decoder
    data_size_bits = 32
    encoder = ArithmeticEncoder(data_size_bits, freq)
    decoder = ArithmeticDecoder(data_size_bits, freq)

    is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder)
    print((encode_len - data_size_bits) / data_block.size, prob_dist.entropy)
    assert is_lossless
