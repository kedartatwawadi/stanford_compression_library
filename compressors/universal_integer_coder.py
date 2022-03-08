"""Simple Universal uint encoder

We implement a very simple universal uint encoder.
Sample encodings:
    0 -> 100
    1 -> 101
    2 -> 11010
    3 -> 11011
    4 -> 1110100 (1110 + 100)
    ...

Encoding: 
1. for encoding x -> get binary code of x, lets call it B[x] (For example: 5 = 101)
2. Encode len(B[x]) as unary ( eg: 4 -> 1110). 
3. The final encode is Unary(len(B[x])) + B[x]

The decoding is straightforward, as the unary code indicates how many bits further to read and decode
"""

from core.data_block import DataBlock
from utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from utils.test_utils import try_lossless_compression
from compressors.prefix_free_compressors import PrefixFreeEncoder, PrefixFreeDecoder


class UniversalUintEncoder(PrefixFreeEncoder):
    """Universal uint encoding:

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


class UniversalUintDecoder(PrefixFreeDecoder):
    """Universal uint Decoder
    0 -> 100
    1 -> 101
    2 -> 11010
    3 -> 11011
    4 -> 1110100 (1110 + 100)
    ...

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


def test_universal_uint_encode_decode():
    encoder = UniversalUintEncoder()
    decoder = UniversalUintDecoder()

    # create some sample data
    data_list = [0, 0, 1, 3, 4, 100]
    data_block = DataBlock(data_list)

    is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless
