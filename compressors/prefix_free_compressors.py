from core.data_compressor import DataCompressor
from core.data_transformer import (
    BitstringToBitsTransformer,
    CascadeTransformer,
    LookupFuncTransformer,
    BitsParserTransformer,
)
from core.data_block import UintDataBlock
from core.util import bitstring_to_uint, uint_to_bitstring


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
