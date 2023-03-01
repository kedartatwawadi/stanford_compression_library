"""Golomb prefix coding for non-negative integers 

Golomb code with parameter M based on
https://en.wikipedia.org/wiki/Golomb_coding#Simple_algorithm.
If M is power of 2, we simplify to Rice codes with slight
change in logic.

The main idea of Golomb code is to divide the input into quotient by M
and the remainder. The quotient is written out in unary, and remained in
binary. The encoding and decoding of the remainder is as described below.

- Let b = floor(log_2(M)). 
- If M is a power of two, we can use exactly b bits to represent the remainder (Rice code). 
- Otherwise, we use either b bits to represent r or b+1 bits to represent r+2^(b+1)-M. 
- We define a parameter cutoff = 2^(b+1)-M to determine which of the cases we fall into (depending on r < cutoff or not).
- During decoding, we can distinguish between these cases based on reading the first b 
bits of remainder encoding and checking if they are < or >= cutoff. Note that if r >= cutoff,
r+cutoff >= 2*cutoff and hence the first b bits are >= cutoff. 

TODO: use latex formatting?
"""

from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataEncoder, DataDecoder
from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from scl.utils.test_utils import try_lossless_compression
import math
from scl.compressors.prefix_free_compressors import (
    PrefixFreeEncoder,
    PrefixFreeDecoder,
)


class GolombCodeParams:
    """Parameters for Golomb codes

    Set up Golomb code parameters (both for encoding and decoding).
    The parameters are described in the module level documentation.
    """

    def __init__(self, M: int):
        """Golomb code parameter initialization

        Args:
            M (int): Golomb code parameter
        """
        assert M > 0
        self.M = M
        self.b = int(math.floor(math.log2(self.M)))
        self.cutoff = 2 ** (self.b + 1) - self.M
        self.rice_code = self.cutoff == self.M


class GolombUintEncoder(PrefixFreeEncoder):
    """Golomb encoder"""

    def __init__(self, M: int):
        """Initialize Golomb encoder

        Args:
            M (int): Golomb code parameter
        """
        self.params = GolombCodeParams(M)

    def encode_symbol(self, x: int):
        """Encode single integer with Golomb code

        Args:
            x (int): integer to encode

        Returns:
            BitArray: encoding of input integer
        """
        assert x >= 0
        assert isinstance(x, int)

        q = x // self.params.M  # quotient
        r = x % self.params.M  # remainder

        # encode quotient in unary
        quotient_bitarray = BitArray(q * "1" + "0")
        # encode remainder in binary using b bits if r < cutoff,
        # or else encode r + cutoff using b+1 bits
        # This is the https://en.wikipedia.org/wiki/Truncated_binary_encoding
        # For M power of 2 (Rice code, always go with b bits)
        if self.params.rice_code or r < self.params.cutoff:
            remainder_bitarray = uint_to_bitarray(r, bit_width=self.params.b)
        else:
            remainder_bitarray = uint_to_bitarray(
                r + self.params.cutoff, bit_width=self.params.b + 1
            )

        return quotient_bitarray + remainder_bitarray


class GolombUintDecoder(PrefixFreeDecoder):
    """Golomb decoder"""

    def __init__(self, M: int):
        """Initialize Golomb decoder

        Args:
            M (int): Golomb code parameter
        """
        self.params = GolombCodeParams(M)

    def decode_symbol(self, encoded_bitarray: BitArray):
        """Decode single integer with Golomb decoding

        Args:
            encoded_bitarray (BitArray): input bitarray with encoding of >=1 integers

        Returns:
            Tuple[Int, Int]: return decoded integer, number of bits read from input
        """
        # initialize num_bits_consumed
        num_bits_consumed = 0

        # infer the quotient
        while True:
            bit = encoded_bitarray[num_bits_consumed]
            num_bits_consumed += 1
            if bit == 0:
                break

        quotient = num_bits_consumed - 1

        # figure out if we encoded remainder with b or b+1 bits
        # For M power of 2 (Rice code, we always go with b bits since it must be < cutoff)
        remainder_first_b_bits = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + self.params.b]
        )
        num_bits_consumed += self.params.b

        if self.params.rice_code or remainder_first_b_bits < self.params.cutoff:
            remainder = remainder_first_b_bits
        else:
            # read one more bit
            remainder_last_bit = int(encoded_bitarray[num_bits_consumed])
            num_bits_consumed += 1
            remainder = 2 * remainder_first_b_bits + remainder_last_bit - self.params.cutoff

        symbol = self.params.M * quotient + remainder

        return symbol, num_bits_consumed


def test_golomb_encode_decode():
    """Tests for Golomb code, including some tests explicitly checking encoding."""
    # first test with M power of 2
    M = 4  # so b = 2 and cutoff = 4 (cutoff can be ignored for M power of 2 which is just Rice code)
    data_block = DataBlock([0, 1, 4, 102])
    expected_output_bitarray = BitArray("000" + "001" + "1000" + "1" * 25 + "0" + "10")
    is_lossless, _, encoded_bitarray = try_lossless_compression(
        data_block, GolombUintEncoder(M), GolombUintDecoder(M)
    )
    assert is_lossless
    assert encoded_bitarray == expected_output_bitarray

    # test with M not power of 2
    M = 10  # so b = 3 and cutoff = 6
    data_block = DataBlock([2, 7, 26, 102])
    expected_output_bitarray = BitArray("0010" + "01101" + "1101100" + "11111111110010")
    is_lossless, _, encoded_bitarray = try_lossless_compression(
        data_block, GolombUintEncoder(M), GolombUintDecoder(M)
    )
    assert is_lossless
    assert encoded_bitarray == expected_output_bitarray

    # bigger test with M power of 2
    M = 4  # so b = 2 and cutoff = 4 (cutoff can be ignored for M power of 2 which is just Rice code)
    data_block = DataBlock(list(range(1000)))
    is_lossless, _, _ = try_lossless_compression(
        data_block, GolombUintEncoder(M), GolombUintDecoder(M)
    )
    assert is_lossless

    # test with M not power of 2
    M = 10  # so b = 3 and cutoff = 6
    data_block = DataBlock(list(range(1000)))
    is_lossless, _, _ = try_lossless_compression(
        data_block, GolombUintEncoder(M), GolombUintDecoder(M)
    )
    assert is_lossless
