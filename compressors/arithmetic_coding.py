from re import S
import numpy as np
from typing import Tuple
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from core.data_block import DataBlock
from core.prob_dist import Frequencies
from utils.test_utils import get_random_data_block


class ArithmeticEncoder(DataEncoder):
    def __init__(self, freqs: Frequencies, state_bit_width=16):
        super().__init__()
        self.state_bit_width = state_bit_width
        self.freqs = freqs

    def shrink_range(self, s, low, high) -> Tuple[int, int]:
        # shrink range
        rng = high - low
        low = low + (rng * self.freqs.cumulative_freq_dict[s]) // self.freqs.total_freq
        high = low + (rng * self.freqs.frequency(s)) // self.freqs.total_freq
        return (low, high)

    def encode_block(self, data_block: DataBlock):
        # initialize the low and high states
        low = 0
        high = (1 << self.state_bit_width) - 1

        HALF_WIDTH = 1 << (self.state_bit_width - 1)
        QTR_WIDTH = 1 << (self.state_bit_width - 2)
        THREE_QTR_WIDTH = 3 * QTR_WIDTH

        # initialize the output
        encoded_bitarray = BitArray("")

        # initialize counter for mid-range re-adjustments
        num_mid_range_readjust = 0

        print(data_block.data_list)
        # start the encoding
        for s in data_block.data_list:

            # shrink range
            low, high = self.shrink_range(s, low, high)
            print(s, low, high)
            while (high - low) < QTR_WIDTH:
                # case 1
                if (low < HALF_WIDTH) and (high < HALF_WIDTH):

                    # output 1's corresponding to prior mid-range readjustments
                    for _ in range(num_mid_range_readjust):
                        encoded_bitarray.append(1)
                        print("here")
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

                    # output 0
                    encoded_bitarray.append(0)
                    print(encoded_bitarray)

                    # re-adjust the range
                    low = low << 1
                    high = high << 1

                elif (low >= HALF_WIDTH) and (high >= HALF_WIDTH):
                    # output 0's corresponding to prior mid-range readjustments
                    for _ in range(num_mid_range_readjust):
                        encoded_bitarray.append(0)
                        print("here")
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

                    # output 1
                    encoded_bitarray.append(1)
                    print(encoded_bitarray)

                    # re-adjust the range
                    low = (low - HALF_WIDTH) << 1
                    high = (high - HALF_WIDTH) << 1

                # elif (low >= HALF_WIDTH) and (high < THREE_QTR_WIDTH):
                #     # increment the mid-range adjustment counter
                #     num_mid_range_readjust += 1

                #     # re-adjust the range
                #     low = (low - QTR_WIDTH) << 1
                #     high = (high - QTR_WIDTH) << 1

                print(f"readjusted range {low}, {high}")
                # NOTE: other scenarios of (low, high) have (high - low) >= QTR_WIDTH, so the range is large enough

        # finally output bits to represent the (low, high) range.
        # as our previous operations guarantee (high - low) >= QTR_WIDTH, we can safely only output 2 bits and be happy
        encoded_bitarray += uint_to_bitarray(low)[:2]

        return encoded_bitarray


class ArithmeticDecoder(DataDecoder):
    def __init__(self, freqs: Frequencies, state_bit_width=16):
        super().__init__()
        self.state_bit_width = state_bit_width
        self.freqs = freqs

    def shrink_range(self, s, low, high) -> Tuple[int, int]:
        # shrink range
        rng = high - low
        low = low + (rng * self.freqs.cumulative_freq_dict[s]) // self.freqs.total_freq
        high = low + (rng * self.freqs.frequency(s)) // self.freqs.total_freq
        return (low, high)

    def decode_block(self, encoded_bitarray: BitArray):
        print("Decoding")
        # initialize return variables
        decoded_data_list = []
        num_bits_consumed = 0

        # initialize the decoding state
        state = bitarray_to_uint(encoded_bitarray[: self.state_bit_width] + BitArray("0" * 12))
        print(state)
        num_bits_consumed += self.state_bit_width

        # initialize the low and high states
        low = 0
        high = (1 << self.state_bit_width) - 1

        STATE_MASK = 1 << (self.state_bit_width)
        HALF_WIDTH = 1 << (self.state_bit_width - 1)
        QTR_WIDTH = 1 << (self.state_bit_width - 2)
        THREE_QTR_WIDTH = 3 * QTR_WIDTH
        breakpoint()
        while True:
            # decode symbol
            # check if range_end and range_start are both in the same bucket
            rng = high - low
            search_list = (
                low
                + (np.array(list(self.freqs.cumulative_freq_dict.values())) * rng)
                // self.freqs.total_freq
            )
            start_bin = np.searchsorted(search_list, state, side="right") - 1
            breakpoint()
            s = self.freqs.alphabet[start_bin]
            decoded_data_list.append(s)
            print(f"Decoded symbol: {s}")
            print(low, high, "before shrinking")

            # shrink range
            low, high = self.shrink_range(s, low, high)
            print(f"range shrunk {low},{high}")

            while (high - low) < QTR_WIDTH:
                # re-adjust range
                # case 1
                if (low < HALF_WIDTH) and (high < HALF_WIDTH):
                    # re-adjust the range
                    low = low << 1
                    high = high << 1

                elif (low >= HALF_WIDTH) and (high >= HALF_WIDTH):

                    # re-adjust the range
                    low = (low - HALF_WIDTH) << 1
                    high = (high - HALF_WIDTH) << 1

                # elif (low >= HALF_WIDTH) and (high < THREE_QTR_WIDTH):
                #     # re-adjust the range
                #     low = (low - QTR_WIDTH) << 1
                #     high = (high - QTR_WIDTH) << 1

                # during the encoding, we had output a 0 to signal this range re-adjustment
                # we can discard this bit now that the re-adjustment is done, and take in the next bit
                if num_bits_consumed >= (len(encoded_bitarray) - 1):
                    bit = 0
                else:
                    num_bits_consumed += 1
                    bit = encoded_bitarray[num_bits_consumed]
                state = ((state << 1) + bit) & STATE_MASK

                print(f"readjusted range: {low}, {high}")
                breakpoint()

            if len(decoded_data_list) == 4:
                return decoded_data_list, num_bits_consumed


def test_arithmetic_coding():
    freq = Frequencies({"A": 6, "B": 4})
    prob_dist = freq.get_prob_dist()

    # generate random data
    data_block = get_random_data_block(prob_dist, 4, seed=0)

    # create encoder decoder
    encoder = ArithmeticEncoder(freq)
    decoder = ArithmeticDecoder(freq)

    encoded_bitarray = encoder.encode_block(data_block)
    breakpoint()
    decoded_symbols, _ = decoder.decode_block(encoded_bitarray)
    breakpoint()
