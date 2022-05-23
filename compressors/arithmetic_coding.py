from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies, get_mean_log_prob
from utils.test_utils import lossless_entropy_coder_test
import abc
import copy


@dataclass
class AECParams:
    """AEC hyper parameters

    These are a couple of parameters used by the AEC encoder. more details inline
    """

    # represents the number of bits used to represent the size of the input data_block
    DATA_BLOCK_SIZE_BITS: int = 32

    # number of bits used to represent the arithmetic coder range
    PRECISION: int = 32

    def __post_init__(self):
        self.FULL: int = 1 << self.PRECISION
        self.HALF: int = 1 << (self.PRECISION - 1)
        self.QTR: int = 1 << (self.PRECISION - 2)
        self.MAX_ALLOWED_TOTAL_FREQ: int = self.QTR
        self.MAX_BLOCK_SIZE: int = 1 << self.DATA_BLOCK_SIZE_BITS


class FreqModelBase(abc.ABC):
    """Base Freq Model

    The AEC encoder can be thought of consisting of two parts:
    1. The probability model
    2. The "lossless coding" algorithm which uses these probabilities

    Note that the probabilities/frequencies coming from the probability model are fixed in the simplest Arithmetic coding version,
    but they can be modified as we parse each symbol.
    This class represents a generic "probability Model", but using frequencies (and hence the name FreqModel).
    (Frequencies are used, mainly because floating point values can be unpredictable/uncertain on different platforms.)

    Some typical examples of Freq models are:

    a) FixedFreqModel -> the probability model is fixed to the initially provided one and does not change
    b) AdaptiveIIDFreqModel -> starts with some initial probability distribution provided (the initial distribution is typically uniform)
       The Adaptive Model then updates the model based on counts of the symbols it sees


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
        appropriately. (see examples below)
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


class ArithmeticEncoder(DataEncoder):
    """Finite precision Arithmetic encoder

    The encoder are decoders are based on the following sources:
    - https://youtu.be/ouYV3rBtrTI: This series of videos on Arithmetic coding are a very gradual but a great
    way to understand them
    - Charles Bloom's blog: https://www.cbloom.com/algs/statisti.html#A5
    - There is of course the original paper: https://web.stanford.edu/class/ee398a/handouts/papers/WittenACM87ArithmCoding.pdf
    """

    def __init__(self, params: AECParams, freq_base: Frequencies, freq_model_cls: FreqModelBase):
        self.params = params

        # define the probability model used by the AEC
        # the model can get updated when we call update_model(s) after every step
        self.freq_model = freq_model_cls(freq_base, params.MAX_ALLOWED_TOTAL_FREQ)

    @classmethod
    def shrink_range(cls, freqs: Frequencies, s: Any, low: int, high: int) -> Tuple[int, int]:
        """shrinks the range (low, high) based on the symbol s

        Args:
            s (Any): symbol to encode

        Returns:
            Tuple[int, int]: (low, high) ranges returned after shrinking
        """

        # compute some intermediate variables: rng, c, d
        rng = high - low
        c = freqs.cumulative_freq_dict[s]
        d = c + freqs.frequency(s)

        # perform shrinking of low, high
        # NOTE: this is the basic Arithmetic coding step implemented using integers
        high = low + (rng * d) // freqs.total_freq
        low = low + (rng * c) // freqs.total_freq
        return (low, high)

    def encode_block(self, data_block: DataBlock):
        """Encode block function for arithmetic coding"""

        # ensure data_block.size is not too big
        err_msg = "choose a larget DATA_BLOCK_SIZE_BITS, as data_block.size is too big"
        assert data_block.size < (1 << self.params.MAX_BLOCK_SIZE), err_msg

        # initialize the low and high states
        low = 0
        high = self.params.FULL

        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning
        # NOTE: Arithmetic decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        encoded_bitarray = uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS)

        # initialize counter for mid-range re-adjustments
        num_mid_range_readjust = 0

        # start the encoding
        for s in data_block.data_list:

            # ensure freqs.total_freq is not too big
            err_msg = """the frequency total is too large, which might cause stability issues. 
            Please increase the precision (or reduce the total_freq"""
            assert (
                self.freq_model.freqs_current.total_freq < self.params.MAX_ALLOWED_TOTAL_FREQ
            ), err_msg

            # shrink range
            # i.e. the core Arithmetic encoding step
            low, high = ArithmeticEncoder.shrink_range(self.freq_model.freqs_current, s, low, high)
            ## update the freq model for encoding the next symbol
            self.freq_model.update_model(s)

            # perform re-normalizing range
            # NOTE: the low, high values need to be re-normalized as else they will keep shrinking
            # and after a few iterations things will be infeasible.
            # The goal of re-normalizing is to not let the range (high - low) get smaller than self.params.QTR

            # CASE I, II -> simple cases where low, high are both in the same half
            while (high < self.params.HALF) or (low > self.params.HALF):
                if high < self.params.HALF:
                    # output 1's corresponding to prior mid-range readjustments
                    encoded_bitarray.extend("0" + "1" * num_mid_range_readjust)

                    # re-adjust range, and reset params
                    low = low << 1
                    high = high << 1
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

                elif low > self.params.HALF:
                    # output 0's corresponding to prior mid-range readjustments
                    encoded_bitarray.extend("1" + "0" * num_mid_range_readjust)

                    # re-adjust range, and reset params
                    low = (low - self.params.HALF) << 1
                    high = (high - self.params.HALF) << 1
                    num_mid_range_readjust = 0  # reset the mid-range readjustment counter

            # CASE III -> the more complex case where low, high straddle the midpoint
            while (low > self.params.QTR) and (high < 3 * self.params.QTR):
                # increment the mid-range adjustment counter
                num_mid_range_readjust += 1
                low = (low - self.params.QTR) << 1
                high = (high - self.params.QTR) << 1

        # Finally output a few bits to signal the final range + any remaining mid range readjustments
        num_mid_range_readjust += 1  # this increment is mainly to output either 01, 10
        if low <= self.params.QTR:
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
    """

    def __init__(self, params: AECParams, freq_base: Frequencies, freq_model_cls: FreqModelBase):
        self.params = params
        self.freq_model = freq_model_cls(freq_base, params.MAX_ALLOWED_TOTAL_FREQ)

    def decode_step_core(self, low: int, high: int, state: int, freqs: Frequencies):
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
            low + (np.array(list(freqs.cumulative_freq_dict.values())) * rng) // freqs.total_freq
        )
        start_bin = np.searchsorted(search_list, state, side="right") - 1
        s = freqs.alphabet[start_bin]
        return s

    def decode_block(self, encoded_bitarray: BitArray):

        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        encoded_bitarray = encoded_bitarray[self.params.DATA_BLOCK_SIZE_BITS :]

        # get data size
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)

        # initialize return variables
        decoded_data_list = []
        num_bits_consumed = 0

        # initialize intermediate state vars etc.
        low = 0
        high = self.params.FULL
        state = 0
        arith_bitarray_size = len(encoded_bitarray)

        # initialize the state
        while (num_bits_consumed < self.params.PRECISION) and (
            num_bits_consumed < arith_bitarray_size
        ):
            bit = encoded_bitarray[num_bits_consumed]
            if bit:
                state += 1 << (self.params.PRECISION - num_bits_consumed - 1)
            num_bits_consumed += 1
        num_bits_consumed = self.params.PRECISION

        # main decoding loop
        while True:
            # decode the next symbol
            s = self.decode_step_core(low, high, state, self.freq_model.freqs_current)
            low, high = ArithmeticEncoder.shrink_range(self.freq_model.freqs_current, s, low, high)
            decoded_data_list.append(s)

            # update the freq_model
            self.freq_model.update_model(s)

            # break when we have decoded all the symbols in the data block
            if len(decoded_data_list) == input_data_block_size:
                break

            while (high < self.params.HALF) or (low > self.params.HALF):
                if high < self.params.HALF:
                    # re-adjust range, and reset params
                    low = low << 1
                    high = high << 1
                    state = state << 1

                elif low > self.params.HALF:
                    # re-adjust range, and reset params
                    low = (low - self.params.HALF) << 1
                    high = (high - self.params.HALF) << 1
                    state = (state - self.params.HALF) << 1

                if num_bits_consumed < arith_bitarray_size:
                    bit = encoded_bitarray[num_bits_consumed]
                    state += bit
                num_bits_consumed += 1

            while (low > self.params.QTR) and (high < 3 * self.params.QTR):
                # increment the mid-range adjustment counter
                low = (low - self.params.QTR) << 1
                high = (high - self.params.QTR) << 1
                state = (state - self.params.QTR) << 1

                if num_bits_consumed < arith_bitarray_size:
                    bit = encoded_bitarray[num_bits_consumed]
                    state += bit
                num_bits_consumed += 1

        # # NOTE: we might have loaded in additional bits not added by the arithmetic encoder
        # # (which are present in the encoded_bitarray).
        # # This block of code determines the extra bits and subtracts it from num_bits_consumed
        for extra_bits_read in range(self.params.PRECISION):
            state_low = (state >> extra_bits_read) << extra_bits_read
            state_high = state_low + (1 << extra_bits_read)
            if (state_low < low) or (state_high > high):
                break
        num_bits_consumed -= extra_bits_read - 1

        # add back the bits corresponding to the num elements
        num_bits_consumed += self.params.DATA_BLOCK_SIZE_BITS

        return DataBlock(decoded_data_list), num_bits_consumed


def test_bitarray_for_specific_input():
    """
    manually perform the encoding and check if the bitarray matches
    """
    # TODO: Kedar


def test_arithmetic_coding():
    """
    Test if AEC coding is working as expcted for different parameter settings
    - Check if encoding/decodng is lossless
    - Check if the compression is close to optimal
    """

    # trying out some random frequencies
    data_freqs_list = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5}),
    ]

    params_list = [
        AECParams(),
        AECParams(),
        AECParams(DATA_BLOCK_SIZE_BITS=12),
        AECParams(DATA_BLOCK_SIZE_BITS=12, PRECISION=16),
    ]

    DATA_SIZE = 1000
    for freq, params in zip(data_freqs_list, params_list):
        # create encoder/decoder
        encoder = ArithmeticEncoder(params, freq, FixedFreqModel)
        decoder = ArithmeticDecoder(params, freq, FixedFreqModel)
        lossless_entropy_coder_test(
            encoder, decoder, freq, DATA_SIZE, encoding_optimality_precision=1e-1, seed=0
        )

    ## create adaptive coder
    for freq, params in zip(data_freqs_list, params_list):
        # create encoder/decoder
        encoder = ArithmeticEncoder(
            params, Frequencies({a: 1 for a in freq.alphabet}), AdaptiveIIDFreqModel
        )
        decoder = ArithmeticDecoder(
            params, Frequencies({a: 1 for a in freq.alphabet}), AdaptiveIIDFreqModel
        )
        lossless_entropy_coder_test(
            encoder, decoder, freq, DATA_SIZE, encoding_optimality_precision=1e-1, seed=0
        )
