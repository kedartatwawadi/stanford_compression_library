"""Elias Delta Universal uint encoder

https://en.wikipedia.org/wiki/Elias_delta_coding

Idea:
To encode X >=0 :
1. Let Y = X+1 so Y > 0
2. Let N = floor(log_2 (Y)) [this is 1 less than the number of bits in binary repr of Y]. Note N>=0
3. Let M = N+1 so M > 0
4. Let L = floor(log_2 (M)) [this is 1 less than the number of bits in binary repr of M].
5. Write L zeros, followed by
6. The L+1 bit binary representation of M(=N+1) [note this always begins with a 1], followed by
7. All but the leading bit of Y(=X+1) (N bits) [leading bit is always 1 so we skip]

Decoding is quite simple:
- first we read the value of L by finding the number of 0s before a 1
- then we read L+1 bits to compute M
- read next N=M-1 bits, prepend a 1 and then decode as an integer

To encode an unsigned integer X, this takes floor(log_2 (X+1)) + 2*floor(log_2(floor(log_2 (X+1)) + 1)) + 1.

Examples:
0 -> 1
1 -> 0 10 0
2 -> 0 10 1
3 -> 0 11 00
...

"""

from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint, BitArray
from scl.utils.test_utils import are_blocks_equal, try_lossless_compression


class EliasDeltaUintEncoder(DataEncoder):
    """See module level documentation"""

    def encode_symbol(self, x: int):
        assert isinstance(x, int)
        assert x >= 0
        y = x + 1
        y_bitarray = uint_to_bitarray(y)
        n = len(y_bitarray) - 1
        m = n + 1
        m_bitarray = uint_to_bitarray(m)
        l = len(m_bitarray) - 1
        return BitArray(l * "0") + m_bitarray + y_bitarray[1:]

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


class EliasDeltaUintDecoder(DataDecoder):
    """See module level documentation"""

    def decode_symbol(self, encoded_bitarray):

        # initialize num_bits_consumed
        num_bits_consumed = 0

        # get the value of L
        while True:
            bit = encoded_bitarray[num_bits_consumed]
            if bit == 1:
                break
            num_bits_consumed += 1

        l = num_bits_consumed

        # decode m
        m = bitarray_to_uint(encoded_bitarray[num_bits_consumed : num_bits_consumed + (l + 1)])
        n = m - 1
        num_bits_consumed += l + 1
        if n == 0:
            y = 1
        else:
            y = bitarray_to_uint(
                BitArray("1") + encoded_bitarray[num_bits_consumed : num_bits_consumed + n]
            )
        num_bits_consumed += n
        x = y - 1
        return x, num_bits_consumed

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


def test_elias_delta_uint_encode_decode():
    """
    Test if the encoding decoding are lossless
    """
    encoder = EliasDeltaUintEncoder()
    decoder = EliasDeltaUintDecoder()

    # create some sample data
    data_list = [0, 0, 1, 3, 0, 0, 0, 2, 1, 4, 100]
    data_block = DataBlock(data_list)

    is_lossless, _, _ = try_lossless_compression(
        data_block,
        encoder,
        decoder,
    )

    assert is_lossless


def test_elias_delta_uint_encode():
    """
    Test if we can recover the expected bitstream
    """
    encoder = EliasDeltaUintEncoder()

    # create some sample data
    data_list = [0, 1, 3, 4, 5, 100]

    # ensure you provide expected codewords for each unique symbol in data_list
    expected_codewords = {
        0: BitArray("1"),
        1: BitArray("0" + "10" + "0"),
        3: BitArray("0" + "11" + "00"),
        4: BitArray("0" + "11" + "01"),
        5: BitArray("0" + "11" + "10"),
        100: BitArray("00" + "111" + "100101"),
    }

    for uint in data_list:
        assert (
            expected_codewords[uint] is not None
        ), "Provide expected codeword for each unique symbol"
        encoded_bitarray = encoder.encode_symbol(uint)
        assert (
            encoded_bitarray == expected_codewords[uint]
        ), "Encoded bitarray does not match expected codeword"
