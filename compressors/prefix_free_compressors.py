from core.data_compressor import DataCompressor
from core.data_transformer import (
    BitstringToBitsTransformer,
    CascadeTransformer,
    LookupFuncTransformer,
    BitsParserTransformer,
)
from core.data_block import UintDataBlock
from core.util import bitstring_to_uint, uint_to_bitstring
import math


class UniversalUintCompressor(DataCompressor):
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

    @staticmethod
    def encoder_lookup_func(x: int):
        assert x >= 0
        assert isinstance(x, int)

        bitstring = uint_to_bitstring(x)
        len_bitstring = len(bitstring) * "1" + "0"

        return len_bitstring + bitstring

    @staticmethod
    def decoder_bits_parser(data_block, start_ind):

        # infer the length
        num_ones = 0
        for ind in range(start_ind, data_block.size):
            bit = data_block.data_list[ind]
            if str(bit) == "0":
                break
            num_ones += 1

        # compute the new start_ind
        new_start_ind = 2 * num_ones + 1 + start_ind

        # decode the symbol
        bitstring = "".join(data_block.data_list[start_ind + num_ones + 1 : new_start_ind])
        symbol = bitstring_to_uint(bitstring)

        return symbol, new_start_ind

    def set_encoder_decoder_params(self, data_block):

        assert isinstance(data_block, UintDataBlock)

        # create encoder and decoder transforms
        self.encoder_transform = CascadeTransformer(
            [
                LookupFuncTransformer(self.encoder_lookup_func),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
        self.decoder_transform = BitsParserTransformer(self.decoder_bits_parser)


class GolombUintCompressor(DataCompressor):
    """
    Golomb code with parameter M based on
    https://en.wikipedia.org/wiki/Golomb_coding#Simple_algorithm:
    If M is power of 2, we simplify to Rice codes with slight
    change in logic.
    """

    def __init__(self, M: int):
        assert M > 0
        self.M = M
        self.b = int(math.floor(math.log2(self.M)))
        self.cutoff = 2 ** (self.b + 1) - self.M
        if self.cutoff == self.M:
            self.rice_code = True
        else:
            self.rice_code = False

    def encoder_lookup_func(self, x: int):
        assert x >= 0
        assert isinstance(x, int)

        q = x // self.M  # quotient
        r = x % self.M  # remainder

        # encode quotient in unary
        quotient_bitstring = q * "1" + "0"
        # encode remainder in binary using b bits if r < cutoff,
        # or else encode r + cutoff using b+1 bits
        # This is the https://en.wikipedia.org/wiki/Truncated_binary_encoding
        # For M power of 2 (Rice code, always go with b bits)
        if self.rice_code or r < self.cutoff:
            remainder_bitstring = uint_to_bitstring(r, bit_width=self.b)
        else:
            remainder_bitstring = uint_to_bitstring(r + self.cutoff, bit_width=self.b + 1)

        return quotient_bitstring + remainder_bitstring

    def decoder_bits_parser(self, data_stream, start_ind):

        # infer the quotient
        quotient = 0
        for ind in range(start_ind, data_stream.size):
            bit = data_stream.data_list[ind]
            if str(bit) == "0":
                break
            quotient += 1

        current_ind = start_ind + quotient + 1

        # see if next bit is 0 or 1 to figure out if we encoded remainder with b or b+1 bits
        # For M power of 2 (Rice code, always go with b bits)
        if self.rice_code or str(data_stream.data_list[current_ind]) == "0":
            new_start_ind = current_ind + self.b
            remainder_bitstring = "".join(data_stream.data_list[current_ind:new_start_ind])
            remainder = bitstring_to_uint(remainder_bitstring)
        else:
            new_start_ind = current_ind + self.b + 1
            remainder_bitstring = "".join(data_stream.data_list[current_ind:new_start_ind])
            remainder = bitstring_to_uint(remainder_bitstring) - self.cutoff

        symbol = self.M * quotient + remainder

        return symbol, new_start_ind

    def set_encoder_decoder_params(self, data_stream):

        assert isinstance(data_stream, UintDataStream)

        # create encoder and decoder transforms
        self.encoder_transform = CascadeTransformer(
            [
                LookupFuncTransformer(self.encoder_lookup_func),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
        self.decoder_transform = BitsParserTransformer(self.decoder_bits_parser)
