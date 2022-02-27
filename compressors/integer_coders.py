from core.data_block import DataBlock
from core.data_encoder_decoder import DataEncoder, DataDecoder
from utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from utils.test_utils import try_lossless_compression
import math


class UniversalUintEncoder(DataEncoder):
    """
    Universal Encoding:
    0 -> 100
    1 -> 101
    2 -> 11010
    3 -> 11011
    4 -> 1110100 (1110 + 100)
    ...
    NOTE: not the most efficient but still "universal"
    """

    def encode_symbol(self, x: int):
        assert isinstance(x, int)
        assert x >= 0

        symbol_bitarray = uint_to_bitarray(x)
        len_bitarray = BitArray(len(symbol_bitarray) * "1" + "0")
        return len_bitarray + symbol_bitarray

    def encode_block(self, data_block: DataBlock):
        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class UniversalUintDecoder(DataDecoder):
    """
    Universal Encoding:
    0 -> 100
    1 -> 101
    2 -> 11010
    3 -> 11011
    4 -> 1110100 (1110 + 100)
    ...
    NOTE: not the most efficient but still "universal"
    """

    def decode_symbol(self, encoded_bitarray):

        # initialize num_bits_consumed
        num_bits_consumed = 0

        # get the symbol length
        while True:
            bit = encoded_bitarray[num_bits_consumed]
            num_bits_consumed += 1
            if bit == 0:
                break
        num_ones = num_bits_consumed - 1

        # decode the symbol
        symbol = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + num_ones]
        )
        num_bits_consumed += num_ones

        return symbol, num_bits_consumed

    def decode_block(self, bitarray: BitArray):
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed


def test_universal_uint_encode_decode():
    encoder = UniversalUintEncoder()
    decoder = UniversalUintDecoder()

    # create some sample data
    data_list = [0, 0, 1, 3, 4, 100]
    data_block = DataBlock(data_list)

    is_lossless, _ = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless


"""
shared initialization for Golomb encoder and decoder
"""


def _initialize_golomb_coder(coder, M: int):
    assert M > 0
    coder.M = M
    coder.b = int(math.floor(math.log2(coder.M)))
    coder.cutoff = 2 ** (coder.b + 1) - coder.M
    if coder.cutoff == coder.M:
        coder.rice_code = True
    else:
        coder.rice_code = False


class GolombUintEncoder(DataEncoder):
    """
    Golomb code with parameter M based on
    https://en.wikipedia.org/wiki/Golomb_coding#Simple_algorithm:
    If M is power of 2, we simplify to Rice codes with slight
    change in logic.
    """

    def __init__(self, M: int):
        _initialize_golomb_coder(self, M)

    def encode_symbol(self, x: int):
        assert x >= 0
        assert isinstance(x, int)

        q = x // self.M  # quotient
        r = x % self.M  # remainder

        # encode quotient in unary
        quotient_bitarray = BitArray(q * "1" + "0")
        # encode remainder in binary using b bits if r < cutoff,
        # or else encode r + cutoff using b+1 bits
        # This is the https://en.wikipedia.org/wiki/Truncated_binary_encoding
        # For M power of 2 (Rice code, always go with b bits)
        if self.rice_code or r < self.cutoff:
            remainder_bitarray = uint_to_bitarray(r, bit_width=self.b)
        else:
            remainder_bitarray = uint_to_bitarray(r + self.cutoff, bit_width=self.b + 1)

        return quotient_bitarray + remainder_bitarray

    def encode_block(self, data_block: DataBlock):
        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class GolombUintDecoder(DataDecoder):
    """
    Golomb code with parameter M based on
    https://en.wikipedia.org/wiki/Golomb_coding#Simple_algorithm:
    If M is power of 2, we simplify to Rice codes with slight
    change in logic.
    """

    def __init__(self, M: int):
        _initialize_golomb_coder(self, M)

    def decode_symbol(self, encoded_bitarray):

        # initialize num_bits_consumed
        num_bits_consumed = 0

        # infer the quotient
        while True:
            bit = encoded_bitarray[num_bits_consumed]
            num_bits_consumed += 1
            if bit == 0:
                break

        quotient = num_bits_consumed - 1

        # see if next bit is 0 or 1 to figure out if we encoded remainder with b or b+1 bits
        # For M power of 2 (Rice code, always go with b bits)
        if self.rice_code or str(encoded_bitarray[num_bits_consumed]) == "0":
            num_bits_remainder = self.b
            remainder = bitarray_to_uint(
                encoded_bitarray[num_bits_consumed : num_bits_consumed + num_bits_remainder]
            )
            num_bits_consumed += num_bits_remainder
        else:
            num_bits_remainder = self.b + 1
            remainder = (
                bitarray_to_uint(
                    encoded_bitarray[num_bits_consumed : num_bits_consumed + num_bits_remainder]
                )
                - self.cutoff
            )
            num_bits_consumed += num_bits_remainder

        symbol = self.M * quotient + remainder

        return symbol, num_bits_consumed

    def decode_block(self, bitarray: BitArray):
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed


def _test_golomb_encode_decode_helper(M, data_list, expected_output_bitarray):
    encoder = GolombUintEncoder(M)
    decoder = GolombUintDecoder(M)

    # sample data
    data_block = DataBlock(data_list)

    # test encode
    encoded_bitarray = encoder.encode_block(data_block)

    assert encoded_bitarray == expected_output_bitarray

    # test decode
    decoded_block, _ = decoder.decode_block(encoded_bitarray)

    # check if the encoding/decoding was lossless
    for inp_symbol, out_symbol in zip(data_block.data_list, decoded_block.data_list):
        assert inp_symbol == out_symbol


def test_golomb_encode_decode():
    # first test with M power of 2
    M = 4  # so b = 2 and cutoff = 4 (cutoff can be ignored for M power of 2 which is just Rice code)
    data_list = [0, 1, 4, 102]
    expected_output_bitarray = BitArray("000" + "001" + "1000" + "1" * 25 + "0" + "10")
    _test_golomb_encode_decode_helper(M, data_list, expected_output_bitarray)

    # test with M not power of 2
    M = 10  # so b = 3 and cutoff = 6
    data_list = [2, 7, 26, 102]
    expected_output_bitarray = BitArray("0010" + "01101" + "1101100" + "11111111110010")
    _test_golomb_encode_decode_helper(M, data_list, expected_output_bitarray)
