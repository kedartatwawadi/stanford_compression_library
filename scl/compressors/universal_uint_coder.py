"""Simple Universal uint encoder

We implement a very simple universal uint encoder.
Sample encodings:
    0 -> 00
    1 -> 01
    2 -> 1010
    3 -> 1011
    4 -> 110100 (110 + 100)
    ...

Encoding: 
1. for encoding x -> get binary code of x, lets call it B[x] (For example: 5 = 101)
2. Encode len(B[x]) as unary (eg: 1->0, 2->10, 3 -> 110).
3. The final encode is Unary(len(B[x])) + B[x]

The decoding is straightforward, as the unary code indicates how many bits further to read and decode
"""

from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from scl.utils.test_utils import are_blocks_equal


class UniversalUintEncoder(DataEncoder):
    """Universal uint encoding:

    0 -> 00
    1 -> 01
    2 -> 1010
    3 -> 1011
    4 -> 110100 (110 + 100)
    ...

    NOTE: not the most efficient but still "universal"
    i.e. works for all symbols
    """

    def encode_symbol(self, x: int):
        assert isinstance(x, int)
        assert x >= 0

        symbol_bitarray = uint_to_bitarray(x)
        len_bitarray = BitArray((len(symbol_bitarray) - 1) * "1" + "0")
        return len_bitarray + symbol_bitarray

    def encode_block(self, data_block: DataBlock) -> BitArray:
        """
        encode the block of data one symbol at a time

        Args:
            data_block (DataBlock): input block to encoded
        Returns:
            BitArray: encoded bitarray
        """

        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class UniversalUintDecoder(DataDecoder):
    """Universal uint Decoder
    00 -> 0
    01 -> 1
    1010 -> 2
    1011 -> 3
    110100 (110 + 100) -> 4
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
        num_ones = num_bits_consumed

        # decode the symbol
        symbol = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + num_ones]
        )
        num_bits_consumed += num_ones

        return symbol, num_bits_consumed

    def decode_block(self, bitarray: BitArray):
        """
        decode the bitarray one symbol at a time using the decode_symbol

        Args:
            bitarray (BitArray): input bitarray with encoding of >=1 integers

        Returns:
            Tuple[DataBlock, Int]: return decoded integers in data block, number of bits read from input
        """
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed


def test_universal_uint_encode_decode():
    """
    Test if the encoding decoding are lossless
    """
    encoder = UniversalUintEncoder()
    decoder = UniversalUintDecoder()

    # create some sample data
    data_list = [0, 0, 1, 3, 4, 100]
    data_block = DataBlock(data_list)

    # test encode
    encoded_bitarray = encoder.encode_block(data_block)

    # test decode
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
    assert num_bits_consumed == len(encoded_bitarray), "Decoder did not consume all bits"

    # compare blocks, and check if the encoding is lossless
    assert are_blocks_equal(
        data_block, decoded_block
    ), "Decoded block does not match original block"

    # run tests


def test_universal_uint_encode():
    """
    Test if the encoded_bitstream matches what we expect
    """
    encoder = UniversalUintEncoder()

    # create some sample data
    data_list = [0, 1, 3, 4, 100]

    # ensure you provide expected codewords for each unique symbol in data_list
    expected_codewords = {
        0: BitArray("00"),
        1: BitArray("01"),
        3: BitArray("1011"),
        4: BitArray("110100"),
        100: BitArray("11111101100100"),
    }

    for uint in data_list:
        assert (
            expected_codewords[uint] is not None
        ), "Provide expected codeword for each unique symbol"
        encoded_bitarray = encoder.encode_symbol(uint)
        assert (
            encoded_bitarray == expected_codewords[uint]
        ), "Encoded bitarray does not match expected codeword"
