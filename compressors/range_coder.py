from bitarray import bitarray
from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from core.prob_dist import Frequencies
import numpy as np
from typing import Tuple, Any

from utils.test_utils import get_random_data_block, try_lossless_compression

# TODO - consider working with numpy's sized integer types. I fear Python types obfuscate
# some of the considerations even though we try to implement here as one would do in C.
# In particular, we need to mask everytime.


class RangeEncoderV2(DataEncoder):
    def __init__(self, precision, freqs):
        assert precision % 8 == 0

        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        # constant params
        # TOP is 1 byte below the max precision
        # For 32 bit, top is simply 2^24
        self.TOP = 1 << (precision - 8)
        # BOTTOM is 2 bytes below the max precision
        # For 32 bit, bottom is simply 2^16
        self.BOTTOM = 1 << (precision - 16)
        # mask is like 0xFFFFFFFF for making sure the arithmetic behaves properly
        # when shifting stuff left
        self.MASK = (1 << self.PRECISION) - 1

        self.freqs = freqs
        assert min(self.freqs.freq_dict.values()) > 0
        assert self.freqs.total_freq <= self.BOTTOM
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

    """
    Return updated low and range
    """

    def normalize(self, encoded_bitarray, low, range_):
        # two things to handle here, separately explained below
        while (low ^ (low + range_)) < self.TOP or range_ < self.BOTTOM:
            # if low and low+range have same top byte we can already output
            # because now we know the range is completely contained in this.
            # another way to write this is that the XOR of low and low+range
            # is less than self.TOP (meaning all bits in top byte must be same)
            if (low ^ (low + range_)) < self.TOP:
                # write the top byte to encoded_bitarray
                encoded_bitarray.frombytes(bytes([low >> (self.PRECISION - 8)]))
                # remove top byte from low and from range (note range anyway has top byte 0
                # to satisfy the top byte of low and high being same)
                low <<= 8
                range_ <<= 8
                low &= self.MASK  # avoid python infinite precision arithmetic issues
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
            if range_ < self.BOTTOM:
                # so we are at a point where the range is too small but the
                # top byte of low and high do not match. This means that we have a very small
                # range but it spans two partitions defined by the top byte equality.

                # more precisely (for 32 bit example), this can only happen when low is
                # like 0xXXFFXXXX and high is like 0xXX00XXXX where the top byte of high is
                # one more than the top byte of low. To solve this we set the new range to be
                # (self.MASK-low)&(self.BOTTOM-1) which basically brings down high to the byte boundary.
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


                range_ = (self.MASK + 1 - low) & (
                    self.BOTTOM - 1
                )  # note that due to python's infinite precision integers and +/-handling the normal C code (-low)&0xFFFF doesn't work.
                # write the top byte to encoded_bitarray
                encoded_bitarray.frombytes(bytes([low >> (self.PRECISION - 8)]))
                # remove top byte from low and from range
                low <<= 8
                range_ <<= 8
                low &= self.MASK  # avoid python infinite precision arithmetic issues
        return low, range_

    def flush(self, encoded_bitarray, low):
        # push out current low
        for _ in range(self.PRECISION // 8):
            encoded_bitarray.frombytes(bytes([low >> (self.PRECISION - 8)]))
            low <<= 8
            low &= self.MASK  # avoid python infinite precision arithmetic issues

    def encode_block(self, data_block: DataBlock):
        """Encode block function for range encoding v2"""
        # initialize the low and high states
        low = 0
        range_ = self.MASK
        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning (rather than using EOF symbol)
        encoded_bitarray = uint_to_bitarray(data_block.size, self.DATA_BLOCK_SIZE_BITS)

        for s in data_block.data_list:
            # shrink range
            low, range_ = RangeEncoderV2.shrink_range(self.freqs, s, low, range_)
            # normalize (also pushes out bits into output)
            low, range_ = self.normalize(encoded_bitarray, low, range_)
        # flush
        self.flush(encoded_bitarray, low)

        return encoded_bitarray


"""get next byte as int for bitarray and return the int and updated num_bits_consumed
"""


def get_next_uint8(bitarr, num_bits_consumed):
    val = bitarray_to_uint(bitarr[num_bits_consumed : num_bits_consumed + 8])
    num_bits_consumed += 8
    return val, num_bits_consumed


class RangeDecoderV2(DataDecoder):

    """Exact same as that for range encoder. FIXME: combine"""

    def __init__(self, precision, freqs):
        assert precision % 8 == 0

        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        # constant params
        # TOP is 1 byte below the max precision
        # For 32 bit, top is simply 2^24
        self.TOP = 1 << (precision - 8)
        # BOTTOM is 2 bytes below the max precision
        # For 32 bit, bottom is simply 2^16
        self.BOTTOM = 1 << (precision - 16)
        # mask is like 0xFFFFFFFF for making sure the arithmetic behaves properly
        # when shifting stuff left
        self.MASK = (1 << self.PRECISION) - 1

        self.freqs = freqs
        assert min(self.freqs.freq_dict.values()) > 0
        assert self.freqs.total_freq <= self.BOTTOM
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

    """
    Return updated low, range, state and num_bits_consumed.
    We skip comments here that are exactly same as encoder normalize.
    The only difference is that here we read bits into state rather than
    writing bits to output.
    """

    def normalize(self, encoded_bitarray, low, range_, state, num_bits_consumed):
        while (low ^ (low + range_)) < self.TOP or range_ < self.BOTTOM:
            if (low ^ (low + range_)) < self.TOP:
                next_byte, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
                state = (state << 8) | next_byte
                state &= self.MASK  # avoid python infinite precision arithmetic issues
                low <<= 8
                range_ <<= 8
                low &= self.MASK  # avoid python infinite precision arithmetic issues
                continue

            if range_ < self.BOTTOM:
                range_ = (self.MASK + 1 - low) & (
                    self.BOTTOM - 1
                )  # note that due to python's infinite precision integers and +/-handling the normal C code (-low)&0xFFFF doesn't work.
                next_byte, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
                state = (state << 8) | next_byte
                state &= self.MASK  # avoid python infinite precision arithmetic issues
                low <<= 8
                range_ <<= 8
                low &= self.MASK  # avoid python infinite precision arithmetic issues
        return low, range_, state, num_bits_consumed

    def decode_block(self, encoded_bitarray: BitArray):
        data_block_size_bitarray = encoded_bitarray[: self.DATA_BLOCK_SIZE_BITS]
        encoded_bitarray = encoded_bitarray[self.DATA_BLOCK_SIZE_BITS :]

        # get data size
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)

        # initialize return variables
        decoded_data_list = []
        num_bits_consumed = 0

        # initialize intermediate state vars etc.
        low = 0
        range_ = self.MASK
        state = 0

        # first read PRECISION // 8 bytes to get the initial state
        # in terms of tracking number of bytes, this corresponds to the flushed bytes at end
        # of encoding, and each step in the while loop below exactly corresponds to each symbol
        # in encoding and the number of bytes match step by step.
        for _ in range(self.PRECISION // 8):
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
                low, range_ = RangeEncoderV2.shrink_range(self.freqs, s, low, range_)

                # normalize
                low, range_, state, num_bits_consumed = self.normalize(
                    encoded_bitarray, low, range_, state, num_bits_consumed
                )

                # break when we have decoded all the symbols in the data block
                if len(decoded_data_list) == input_data_block_size:
                    break

        # add the bits corresponding to the num elements
        num_bits_consumed += self.DATA_BLOCK_SIZE_BITS

        return DataBlock(decoded_data_list), num_bits_consumed


"""provide custom_input if you want to test edge case rather than probabilistic
"""


def _test_range_coding_v2(freq, data_size, seed, custom_input=None, no_print=False):
    prob_dist = freq.get_prob_dist()

    # generate random data
    if custom_input is None:
        data_block = get_random_data_block(prob_dist, data_size, seed=seed)
    else:
        data_block = DataBlock(custom_input)
    precision = 32
    data_size_bits = 32  # for storing data size in header
    # create encoder decoder
    encoder = RangeEncoderV2(precision, freq)
    decoder = RangeDecoderV2(precision, freq)

    is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder)
    if not no_print:
        print(
            "bits per symbol, entropy:",
            (encode_len - data_size_bits) / data_block.size,
            prob_dist.entropy,
        )

    assert is_lossless


def test_range_coding_v2():
    DATA_SIZE = 5000
    freqs = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 1, "C": 65534}),
    ]

    for freq in freqs:
        _test_range_coding_v2(freq, DATA_SIZE, seed=0)

    _test_range_coding_v2(
        Frequencies({"A": 1, "C": 65535}), None, seed=None, custom_input=["A", "C"] * 5000
    )
    _test_range_coding_v2(
        Frequencies({"A": 1, "B": 1, "C": 65534}),
        None,
        seed=None,
        custom_input=["A", "B", "C"] * 2000,
    )
    _test_range_coding_v2(
        Frequencies({"A": 1, "B": 1, "C": 65534}), None, seed=None, custom_input=["A"] * 5000
    )
    _test_range_coding_v2(
        Frequencies({"A": 1, "B": 1, "C": 65534}), None, seed=None, custom_input=["C"] * 5000
    )

    freq = Frequencies({"A": 12, "B": 34, "C": 1, "D": 45})
    prob_dist = freq.get_prob_dist()
    data_block = get_random_data_block(prob_dist, 5000, seed=0)
    for l in range(0, 50):
        _test_range_coding_v2(
            freq, None, seed=0, custom_input=data_block.data_list[:l], no_print=True
        )


"""
Notes (to be put in wiki).

## RangeCoderV2
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
"""

class RangeEncoderV1(DataEncoder):
    def __init__(self, precision, freqs):
        assert precision % 8 == 0

        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        # constant params
        # TOP is 1 bit below the max precision
        # For 32 bit, top is simply 2^31 or 0x80000000
        self.TOP = 1 << (precision - 1)
        # BOTTOM is 1 bytes below TOP
        # For 32 bit, bottom is simply 2^23 or 0x00800000
        self.BOTTOM = self.TOP >> 8
        # mask is like 0xFFFFFFFF for making sure the arithmetic behaves properly
        # when shifting stuff left
        self.MASK = (1 << self.PRECISION) - 1
        # used during normalization to determine if we need to wait before releasing
        # bytes, and also to get the next buffer from low
        self.SHIFT_BITS = self.PRECISION - 9

        self.freqs = freqs
        assert min(self.freqs.freq_dict.values()) > 0
        assert self.freqs.total_freq <= self.BOTTOM
        super().__init__()


    # TODO: this shrink range is same as V2 shrink range, can be combined
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

    def normalize(self, encoded_bitarray, low, range_, buffer, mid_range_count):
        """
        Return updated low, range, buffer mid_range_count
        """
        
        while range_ <= self.BOTTOM:
            if low < (0xFF << self.SHIFT_BITS):
                encoded_bitarray.frombytes(bytes([buffer]))
                for _ in range(mid_range_count):
                    encoded_bitarray.frombytes(bytes([0xFF]))
                mid_range_count = 0
                buffer = 0xFF & (low >> self.SHIFT_BITS)
            elif low & self.TOP != 0:
                encoded_bitarray.frombytes(bytes([buffer + 1]))
                for _ in range(mid_range_count):
                    encoded_bitarray.frombytes(bytes([0x00]))
                mid_range_count = 0
                buffer = 0xFF & (low >> self.SHIFT_BITS)
            else:
                mid_range_count += 1
            range_ <<= 8
            low = (low<<8) & (self.TOP-1)
        return low, range_, buffer, mid_range_count

    def flush(self, encoded_bitarray, low, range_, buffer, mid_range_count):
        low, range_, buffer, mid_range_count = self.normalize(encoded_bitarray, low, range_, buffer, mid_range_count)
        
        # now we flush from low. For simplicity if we assume that we didn't get mid range
        # in normalize, that means that byte 30-23 from old low has been put in buffer, and
        # new low 0x00 as the lowest byte. So we basically release the buffer now, as well
        # as byte 30-23 & byte 22-15 of current low. TODO: understand why we don't need the
        # byte 14-7 of low. 
        # Of course, we need to be careful while handling the carry part
        # and also need to release any mid range stuff that's accumulated.
        if low & self.TOP != 0:
            # we have a carry
            encoded_bitarray.frombytes(bytes([buffer + 1]))
            for _ in range(mid_range_count):
                encoded_bitarray.frombytes(bytes([0x00]))
        else:
            # no carry
            encoded_bitarray.frombytes(bytes([buffer]))
            for _ in range(mid_range_count):
                encoded_bitarray.frombytes(bytes([0xFF]))
        
        # flush out rest of low apart from the carry bit
        low = (low << 1) & self.MASK
        for _ in range(self.PRECISION // 8):
            encoded_bitarray.frombytes(bytes([low >> (self.PRECISION - 8)]))
            low <<= 8
            low &= self.MASK  # avoid python infinite precision arithmetic issues

    def encode_block(self, data_block: DataBlock):
        """Encode block function for range encoding v2"""
        # initialize the low and high states
        low = 0
        range_ = self.TOP
        mid_range_count = 0 # this is called `help` in the C version
        buffer = 0x00
        # initialize the output
        encoded_bitarray = BitArray("")

        # add the data_block size at the beginning (rather than using EOF symbol)
        encoded_bitarray = uint_to_bitarray(data_block.size, self.DATA_BLOCK_SIZE_BITS)
        for s in data_block.data_list:
            # normalize (also pushes out bits into output)
            low, range_, buffer, mid_range_count = self.normalize(encoded_bitarray, low, range_, buffer, mid_range_count)
            # shrink range
            low, range_ = RangeEncoderV1.shrink_range(self.freqs, s, low, range_)

        # flush
        self.flush(encoded_bitarray, low, range_, buffer, mid_range_count)
        return encoded_bitarray


class RangeDecoderV1(DataDecoder):

    """Very similar as that for range encoder. FIXME: combine"""

    def __init__(self, precision, freqs):
        assert precision % 8 == 0

        self.DATA_BLOCK_SIZE_BITS = 32  # represents the size of the data block
        self.PRECISION = precision
        # constant params
        # TOP is 1 bit below the max precision
        # For 32 bit, top is simply 2^31 or 0x80000000
        self.TOP = 1 << (precision - 1)
        # BOTTOM is 1 bytes below TOP
        # For 32 bit, bottom is simply 2^23 or 0x00800000
        self.BOTTOM = self.TOP >> 8
        # mask is like 0xFFFFFFFF for making sure the arithmetic behaves properly
        # when shifting stuff left
        self.MASK = (1 << self.PRECISION) - 1
        # used during normalization to determine if we need to wait before releasing
        # bytes, and also to get the next buffer from low
        self.SHIFT_BITS = self.PRECISION - 9

        # this is 7 whenever precision is multiple of 8, this relates to the fact that
        # top bit is used for carry during encoding so when we put the read bytes into
        # state, only the top 7 bits go into it initially, and last bit will go in next turn
        self.EXTRA_BITS = (self.PRECISION-2) % 8 + 1

        self.freqs = freqs
        assert min(self.freqs.freq_dict.values()) > 0
        assert self.freqs.total_freq <= self.BOTTOM
        super().__init__()

    def decode_symbol(self, range_: int, state: int):
        """Core range decoding function
        We cut the [0, range_) range bits proportional to the cumulative probability of each symbol
        the function locates the bin in which the state lies
        """

        # FIXME: simplify this search.
        search_list = (
            np.array(list(self.freqs.cumulative_freq_dict.values()))
            * (range_ // self.freqs.total_freq)
        )
        start_bin = np.searchsorted(search_list, state, side="right") - 1
        s = self.freqs.alphabet[start_bin]
        return s

    @classmethod
    def shrink_range(cls, freqs: Frequencies, s: Any, range_: int, state) -> Tuple[int, int]:
        """shrinks the range (0, range) based on the symbol s, also update state accordingly
        Args:
            s (Any): symbol to encode
        Returns:
            Tuple[int, int]: range, state returned after shrinking
        """
        # compute some intermediate variables: c, d
        c = freqs.cumulative_freq_dict[s]
        d = c + freqs.frequency(s)
        # perform shrinking of low, range
        # NOTE: this is the basic range coding step implemented using integers
        range_ = range_ // freqs.total_freq
        state -= c * range_
        range_ *= d - c

        return range_, state

    """
    Return updated low, range, state, buffer and num_bits_consumed.
    """
    def normalize(self, encoded_bitarray, range_, state, buffer, num_bits_consumed):
        while range_ <= self.BOTTOM:
            state = (state << 8) | ((buffer << self.EXTRA_BITS) & 0xFF)
            buffer, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
            state = state | (buffer >> (8-self.EXTRA_BITS))
            state = state & self.MASK
            range_ <<= 8
        return range_, state, buffer, num_bits_consumed

    def decode_block(self, encoded_bitarray: BitArray):
        data_block_size_bitarray = encoded_bitarray[: self.DATA_BLOCK_SIZE_BITS]
        encoded_bitarray = encoded_bitarray[self.DATA_BLOCK_SIZE_BITS :]
        # get data size
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)

        # initialize return variables
        decoded_data_list = []
        num_bits_consumed = 0

        # initialize intermediate state vars etc.
        range_ = 1<<self.EXTRA_BITS # this is simply 0x80 - the first call to normalize
        # will get it to self.TOP which is what we initialized with during encoding
        # We fix low to 0 and just work with range and state to keep things simple

        # consume first byte which is useless (initial buffer put by encoder to simplify logic)
        _, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)
        # consume next byte and use it to initiate state
        buffer, num_bits_consumed = get_next_uint8(encoded_bitarray, num_bits_consumed)

        state = buffer >> (8 - self.EXTRA_BITS)
        if input_data_block_size == 0:
            # empty, so don't enter while loop
            pass
        else:
            while True:
                # first call normalize to get range of correct size (also reads from input)
                range_, state, buffer, num_bits_consumed = self.normalize(
                    encoded_bitarray, range_, state, buffer, num_bits_consumed
                )
                # decode a symbol based on range_ and state
                s = self.decode_symbol(range_, state)
                decoded_data_list.append(s)
                # shrink range (reuse range encoder code)
                range_, state = RangeDecoderV1.shrink_range(self.freqs, s, range_, state)
                # break when we have decoded all the symbols in the data block
                if len(decoded_data_list) == input_data_block_size:
                    break

        # normalize one last time to read any remaining bytes in input
        _,_,_,num_bits_consumed = self.normalize(
                    encoded_bitarray, range_, state, buffer, num_bits_consumed
                )

        # add the bits corresponding to the num elements
        num_bits_consumed += self.DATA_BLOCK_SIZE_BITS
        return DataBlock(decoded_data_list), num_bits_consumed


"""provide custom_input if you want to test edge case rather than probabilistic
"""


def _test_range_coding_v1(freq, data_size, seed, custom_input=None, no_print=False):
    if no_print is False or custom_input is None:
        prob_dist = freq.get_prob_dist()

    # generate random data
    if custom_input is None:
        data_block = get_random_data_block(prob_dist, data_size, seed=seed)
    else:
        data_block = DataBlock(custom_input)
    precision = 32
    data_size_bits = 32  # for storing data size in header
    # create encoder decoder
    encoder = RangeEncoderV1(precision, freq)
    decoder = RangeDecoderV1(precision, freq)

    is_lossless, encode_len, _ = try_lossless_compression(data_block, encoder, decoder)
    if not no_print:
        print(
            "bits per symbol, entropy:",
            (encode_len - data_size_bits) / data_block.size,
            prob_dist.entropy,
        )

    assert is_lossless


def test_range_coding_v1():
    DATA_SIZE = 5000
    freqs = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 12, "B": 34, "C": 1, "D": 45}),
        Frequencies({"A": 34, "B": 35, "C": 546, "D": 1, "E": 13, "F": 245}),
        Frequencies({"A": 1, "C": 65535}),
    ]

    for freq in freqs:
        print(freq)
        _test_range_coding_v1(freq, DATA_SIZE, seed=0)

    _test_range_coding_v1(
        Frequencies({"A": 1, "C": 1<<23-1}), None, seed=None, custom_input=["A", "C"] * 5000,no_print=True
    )
    _test_range_coding_v1(
        Frequencies({"A": 1, "C": 1<<23-1}), None, seed=None, custom_input=["A"] * 5000,no_print=True
    )
    _test_range_coding_v1(
        Frequencies({"A": 1, "C": 1<<23-1}), None, seed=None, custom_input=["C"] * 5000, no_print=True
    )

    _test_range_coding_v1(
        Frequencies({"A": 1, "B": 1, "C": 1<<23-2}),
        None,
        seed=None,
        custom_input=["A", "B", "C"] * 2000,
        no_print=True
    )

    freq = Frequencies({"A": 12, "B": 34, "C": 1, "D": 45})
    prob_dist = freq.get_prob_dist()
    data_block = get_random_data_block(prob_dist, 5000, seed=0)
    for l in range(0, 50):
        _test_range_coding_v2(
            freq, None, seed=0, custom_input=data_block.data_list[:l], no_print=True
        )

"""
Notes (to be put in wiki).

## RangeCoderV1
This is the canonical range coder or Schindler range coder. I believe this is very similar
to the Martin paper: https://sachingarg.com/compression/entropy_coding/range_coder.pdf. It was
a bit hard to read unfortunately so I can't say for sure if there are differences.

The implementation here is based on:
- Arturo Campos's write up and pesudocode at https://web.archive.org/web/20160423141129/http://www.arturocampos.com/ac_range.html
- Schindler's code: https://github.com/makinacorpus/libecw/blob/d5bef9682e05f3b044d7e5a19665efa27ff62c7c/Source/C/NCSEcw/shared_src/rangecode.c

This is more efficient than the Russian range coder since it more carefully manages the mid 
range stuff rather than just wasting part of the range. In addition, this allows us to have
higher frequency precision (23 bit vs. 16 bit when using 32 bit ints). 
It is slightly more complicated however. Below I try to describe the main ideas,
working throughout with 32 bit precision for the sake of exposition.

Let's define the following:
Top value = 0x80000000 = 1000...0000
Bottom value = 0x00800000 [1 byte below top]

As usual, we deal with low and range, where high = low+range. The shrink_range stuff is
exactly same as RangeCoderV2, which is basically same as arithmetic coding, so we skip
that here (similarly for the decode_symbol function).

We use at most 31 bits (30-0) for the range, i.e., range is at most top value. We use
23 bits for the distribution (i.e., sum of frequencies <= bottom value). Once range gets
below this we need to normalize.

The top bit (bit 31) is used as a carry. Say if low (bit 30-0) starts with 

Top bit (31) - “carry”
30-22 - “renormalization byte”
Bit 21 

Top value:       0x80000000
Bottom value: 0x00800000

First written byte is arbitrary to simplify rest of logic slightly

Bottom value is max value of range

Encoder normalize:
While range is less than bottom:
If top bit of low is 1, we can increment previous buffer by 1 release it, and put the next byte (30-22) into buffer
If top bit is 0 and renorm byte is not 0xFF, release previous buffer, and put the renorm byte into buffer
In above two cases we are not sure if the eventual range has the renorm byte or that +1 which is why we are not ready to release it yet.

The last case is similar to mid range for arithmetic coding.
If top bit is 0 and renorm byte is 0xFF, we can’t quite release anything yet
because we are not sure if the top bit will be eventually 0 or 1. Instead we 
increment a counter for number of times we ran into this situation, and later
once we hit one of the previous two cases, we first release (depending on the 
case: either buffer followed by bunch of 0xFFs or buffer+1 followed by bunch of 0x00s).

This is very similar to arithmetic coding approach, except that we deal with a byte rather
than a bit. Another difference is that we don't check high for some reason, rather always
assume there is a possibility of carry happening.

Why is range at most top value? 
Top bit is not used for range, it is only for carry. So we always release bits 30-23 (renorm byte)
into buffer knowing that the we want to either release this byte or this plus 1 (ignoring the last "mid-range" 
case at the moment). Later based on the carry we will realize whether we need to add 1 and only
then we actually put it into the output from the buffer. If we didn't have this bit, we would
have overflow and won't be able to capture the carry.

Note that range (after normalization satisfies): 0x00800000 < range <= 0x80000000.
Since each normalization loop shifts range left by 8 bits, this means that we can be sure
that the normalization step during decoding end up with same range (and hence low) as the
normalization step during encoding (after each symbol).

Correspondence between encoding and decoding states and how does the end flush part
work correctly to guarantee that we read all the bytes we have written:
(stages on same line have same exact state (low, range, buffer, mid range count) after they finish)

Decoding:

C implementation fixes low to be 0 and uses the variable named low to represent
what we call state. We stick with the low, range, state variables as used in V2 and in
arithmetic coding for clarity.

"""

