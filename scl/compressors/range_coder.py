"""
## RangeCoder
Also called "Russian range coder" in certain places, e.g., at
http://cbloomrants.blogspot.com/2008/10/10-05-08-5.html. This is the range coder
used in PPMd and is somewhat less optimal since it just sort of gives up when facing 
underflow and simply releases bytes without trying to figure out if the ultimate range is
above or below mid. Bloom suggests possible optimization for this in 
https://cbloomrants.blogspot.com/2008/10/10-06-08-followup-on-russian-range-coder.html.

We implement here the version as described in https://sachingarg.com/compression/entropy_coding/64bit/
(code at https://sachingarg.com/compression/entropy_coding/64bit/code/entropy/range32.cpp) which
is the standard version as described in Blooms first post above.

The range coder as described on Wikipedia (https://en.wikipedia.org/w/index.php?title=Range_coding&oldid=1069851861)
is very similar to this approach.

https://web.archive.org/web/20020420161153/http://www.softcomplete.com/algo/pack/rus-range.txt

Not completely understood points:
1. Why is bottom chosen as 2^16 (for 32 bit arithmetic)? Note that bottom determines the max sum of 
   frequencies and we do not allow range to go below this. Clearly too high and too low values
   have issues (too frequent normalization/loss vs. too low resolution in probabilities). Is
   2^16 the only value that would work? Is there a significance to it being byte aligned (i.e., 16=2*8)?

   Based on testing, setting bottom to 2^20 also works fine, but the gap to entropy is slightly higher.

## Canonical range coder
Another range coder implementation based on carry is the canonical range coder or Schindler 
range coder (not included here). It is very similar to the Martin paper: 
https://sachingarg.com/compression/entropy_coding/range_coder.pdf. 
Two detailed writeups/implementations for the canonical range
- Arturo Campos's write up and pesudocode at https://web.archive.org/web/20160423141129/http://www.arturocampos.com/ac_range.html
- Schindler's code: https://github.com/makinacorpus/libecw/blob/d5bef9682e05f3b044d7e5a19665efa27ff62c7c/Source/C/NCSEcw/shared_src/rangecode.c

This is more efficient than the Russian range coder since it more carefully manages the mid 
range stuff rather than just wasting part of the range. In addition, this allows us to have
higher frequency precision (23 bit vs. 16 bit when using 32 bit ints). 
"""

from dataclasses import dataclass
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.core.prob_dist import Frequencies
import numpy as np
from typing import Tuple, Any
from scl.utils.test_utils import lossless_entropy_coder_test
from scl.utils.test_utils import get_random_data_block, try_lossless_compression

# TODO - consider working with numpy's sized integer types. I fear Python types obfuscate
# some of the considerations even though we try to implement here as one would do in C.
# In particular, we need to mask everytime.


@dataclass
class RangeCoderParams:
    """Range coder parameters"""

    # represents the number of bits used to represent the size of the input data_block
    DATA_BLOCK_SIZE_BITS: int = 32

    # number of bits used to represent the arithmetic coder range
    PRECISION: int = 32

    def __post_init__(self):
        assert self.PRECISION % 8 == 0
        # constant params
        # TOP is 1 byte below the max precision
        # For 32 bit, top is simply 2^24
        self.TOP = 1 << (self.PRECISION - 8)
        # BOTTOM is 2 bytes below the max precision
        # For 32 bit, bottom is simply 2^16
        self.BOTTOM = 1 << (self.PRECISION - 16)
        # mask is like 0xFFFFFFFF for making sure the arithmetic behaves properly
        # when shifting stuff left
        self.MASK = (1 << self.PRECISION) - 1


class RangeEncoder(DataEncoder):
    def __init__(self, params: RangeCoderParams, freqs: Frequencies):
        self.params = params

        self.freqs = freqs
        assert min(self.freqs.freq_dict.values()) > 0
        assert self.freqs.total_freq <= self.params.BOTTOM
        super().__init__()

    @classmethod
    def shrink_range(cls, freqs: Frequencies, s: Any, low: int, range_: int) -> Tuple[int, int]:
        """shrinks the range (low, low+range) based on the symbol s
        Args:
            s (Any): symbol to encode
        Returns:
            Tuple[int, int]: (low, range) ranges returned after shrinking
        """
        # compute some intermediate variables: c, d
        c = freqs.cumulative_freq_dict[s]
        d = c + freqs.frequency(s)
        # perform shrinking of low, range
        # NOTE: this is the basic range coding step implemented using integers
        range_ = range_ // freqs.total_freq
        low += c * range_
        range_ *= d - c

        return low, range_

    def normalize(self, encoded_bitarray, low, range_):
        """
        Return updated low and range
        """
        # two things to handle here, separately explained below
        while (low ^ (low + range_)) < self.params.TOP or range_ < self.params.BOTTOM:
            # if low and low+range have same top byte we can already output
            # because now we know the range is completely contained in this.
            # another way to write this is that the XOR of low and low+range
            # is less than self.params.TOP (meaning all bits in top byte must be same)
            if (low ^ (low + range_)) < self.params.TOP:
                # write the top byte to encoded_bitarray
                encoded_bitarray.frombytes(bytes([low >> (self.params.PRECISION - 8)]))
                # remove top byte from low and from range (note range anyway has top byte 0
                # to satisfy the top byte of low and high being same)
                low <<= 8
                range_ <<= 8
                low &= self.params.MASK  # avoid python infinite precision arithmetic issues
                continue
                # in terms of range the right way to think about this while loop:
                # Think of the total possible range being divided into 256 partitions where
                # a partition corresponds to a top byte value. Here, we have the range lying
                # completely within one partition so we know the top byte can be safely
                # released and we can just focus on working within the partition with higher
                # resolution.

            # next we try to determine if the range has gotten too small which might
            # lead to underflow and inability to divide it further according to probability
            # distribution
            if range_ < self.params.BOTTOM:
                # so we are at a point where the range is too small but the
                # top byte of low and high do not match. This means that we have a very small
                # range but it spans two partitions defined by the top byte equality.

                # more precisely (for 32 bit example), this can only happen when low is
                # like 0xXXFFXXXX and high is like 0xXX00XXXX where the top byte of high is
                # one more than the top byte of low. To solve this we set the new range to be
                # (self.params.MASK-low)&(self.params.BOTTOM-1) which basically brings down high to the byte boundary.
                # After this we can release a byte and continue with the loop. Some examples below (32 bit case):

                # old_low: 0xabffacdd
                # old_range: 0xfff0
                # old_high: 0xac00accd
                # range_after_step 0x5323
                # high_after_step 0xac000000
                # byte_released 0xab

                # old_low: 0xabffffff
                # old_range: 0x54
                # old_high: 0xac000053
                # range_after_step 0x1
                # high_after_step 0xac000000
                # byte_released 0xab

                # def print_info(low,range):
                #     print("old_low:", hex(low))
                #     print("old_range:", hex(range))
                #     print("old_high:", hex(low+range))
                #     range_after_step = (0xFFFFFFFF-low+1)&0xFFFF
                #     print("range_after_step", hex(range_after_step))
                #     print("high_after_step", hex(low+range_after_step))
                #     print("byte_released", hex(low//int("0xFFFFFF",0))

                range_ = (self.params.MASK + 1 - low) & (
                    self.params.BOTTOM - 1
                )  # note that due to python's infinite precision integers and +/-handling the normal C code (-low)&0xFFFF doesn't work.
                # write the top byte to encoded_bitarray
                encoded_bitarray.frombytes(bytes([low >> (self.params.PRECISION - 8)]))
                # remove top byte from low and from range
                low <<= 8
                range_ <<= 8
                low &= self.params.MASK  # avoid python infinite precision arithmetic issues
        return low, range_

    def flush(self, encoded_bitarray, low):
        # push out current low
        for _ in range(self.params.PRECISION // 8):
            encoded_bitarray.frombytes(bytes([low >> (self.params.PRECISION - 8)]))
            low <<= 8
            low &= self.params.MASK  # avoid python infinite precision arithmetic issues

    def encode_block(self, data_block: DataBlock):
        """Encode block function for range encoding v2"""
        # initialize the low and high states
        low = 0
        range_ = self.params.MASK
        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning (rather than using EOF symbol)
        encoded_bitarray = uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS)

        for s in data_block.data_list:
            # shrink range
            low, range_ = RangeEncoder.shrink_range(self.freqs, s, low, range_)
            # normalize (also pushes out bits into output)
            low, range_ = self.normalize(encoded_bitarray, low, range_)
        # flush
        self.flush(encoded_bitarray, low)

        return encoded_bitarray


def get_next_uint8(bitarr, num_bits_consumed):
    """get next byte as int for bitarray and return the int and updated num_bits_consumed"""
    val = bitarray_to_uint(bitarr[num_bits_consumed : num_bits_consumed + 8])
    num_bits_consumed += 8
    return val, num_bits_consumed


class RangeDecoder(DataDecoder):
    def __init__(self, params: RangeCoderParams, freqs):
        self.params = params
        self.freqs = freqs
        assert min(self.freqs.freq_dict.values()) > 0
        assert self.freqs.total_freq <= self.params.BOTTOM
        super().__init__()

    def decode_symbol(self, low: int, range_: int, state: int):
        """Core range decoding function
        We cut the [low, high) range bits proportional to the cumulative probability of each symbol
        the function locates the bin in which the state lies
        """

        # FIXME: simplify this search.
        search_list = low + (
            np.array(list(self.freqs.cumulative_freq_dict.values()))
            * (range_ // self.freqs.total_freq)
        )
        start_bin = np.searchsorted(search_list, state, side="right") - 1
        s = self.freqs.alphabet[start_bin]
        return s

    def normalize(self, encoded_bitarray, low, range_, state, num_bits_consumed):
        """
        Return updated low, range, state and num_bits_consumed.
        We skip comments here that are exactly same as encoder normalize.
        The only difference is that here we read bits into state rather than
        writing bits to output.
        """
        while (low ^ (low + range_)) < self.params.TOP or range_ < self.params.BOTTOM:
            if (low ^ (low + range_)) < self.params.TOP:
                next_byte, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
                state = (state << 8) | next_byte
                state &= self.params.MASK  # avoid python infinite precision arithmetic issues
                low <<= 8
                range_ <<= 8
                low &= self.params.MASK  # avoid python infinite precision arithmetic issues
                continue

            if range_ < self.params.BOTTOM:
                range_ = (self.params.MASK + 1 - low) & (
                    self.params.BOTTOM - 1
                )  # note that due to python's infinite precision integers and +/-handling the normal C code (-low)&0xFFFF doesn't work.
                next_byte, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
                state = (state << 8) | next_byte
                state &= self.params.MASK  # avoid python infinite precision arithmetic issues
                low <<= 8
                range_ <<= 8
                low &= self.params.MASK  # avoid python infinite precision arithmetic issues
        return low, range_, state, num_bits_consumed

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
        range_ = self.params.MASK
        state = 0

        # first read PRECISION // 8 bytes to get the initial state
        # in terms of tracking number of bytes, this corresponds to the flushed bytes at end
        # of encoding, and each step in the while loop below exactly corresponds to each symbol
        # in encoding and the number of bytes match step by step.
        for _ in range(self.params.PRECISION // 8):
            next_byte, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
            state = (state << 8) | next_byte

        if input_data_block_size == 0:
            # empty, so don't enter while loop
            pass
        else:
            while True:
                # decode a symbol based on low and range_
                s = self.decode_symbol(low, range_, state)
                decoded_data_list.append(s)

                # shrink range (reuse range encoder code)
                low, range_ = RangeEncoder.shrink_range(self.freqs, s, low, range_)

                # normalize
                low, range_, state, num_bits_consumed = self.normalize(
                    encoded_bitarray, low, range_, state, num_bits_consumed
                )

                # break when we have decoded all the symbols in the data block
                if len(decoded_data_list) == input_data_block_size:
                    break

        # add the bits corresponding to the num elements
        num_bits_consumed += self.params.DATA_BLOCK_SIZE_BITS

        return DataBlock(decoded_data_list), num_bits_consumed


def _test_range_coding(freq, input):
    data_block = DataBlock(input)
    # create encoder decoder
    encoder = RangeEncoder(RangeCoderParams(), freq)
    decoder = RangeDecoder(RangeCoderParams(), freq)

    is_lossless, _, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )

    assert is_lossless


def test_range_coding():
    print()
    DATA_SIZE = 10000
    freqs = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 1, "C": 65534}),
    ]

    for freq in freqs:
        # create encoder decoder
        encoder = RangeEncoder(RangeCoderParams(), freq)
        decoder = RangeDecoder(RangeCoderParams(), freq)
        lossless_entropy_coder_test(
            encoder, decoder, freq, DATA_SIZE, encoding_optimality_precision=0.1
        )

    # now test various edge cases and specific inputs
    _test_range_coding(
        Frequencies({"A": 1, "C": 65535}),
        ["A", "C"] * 5000,
    )
    _test_range_coding(
        Frequencies({"A": 1, "B": 1, "C": 65534}),
        ["A", "B", "C"] * 2000,
    )
    _test_range_coding(
        Frequencies({"A": 1, "B": 1, "C": 65534}),
        ["A"] * 5000,
    )
    _test_range_coding(
        Frequencies({"A": 1, "B": 1, "C": 65534}),
        ["C"] * 5000,
    )

    # test various length inputs to ensure everything runs smoothly with the flushing etc.
    freq = Frequencies({"A": 12, "B": 34, "C": 1, "D": 45})
    prob_dist = freq.get_prob_dist()
    data_block = get_random_data_block(prob_dist, 5000, seed=0)
    for l in range(0, 50):
        _test_range_coding(freq, data_block.data_list[:l])
