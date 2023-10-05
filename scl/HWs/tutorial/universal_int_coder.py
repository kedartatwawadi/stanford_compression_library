"""Simple Universal Integer coder

Extends `compressors/universal_uint_coder.py` to handle signed integers
"""

from core.data_block import DataBlock
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray
from utils.test_utils import are_blocks_equal
from compressors.universal_uint_coder import UniversalUintEncoder, UniversalUintDecoder


class UniversalIntegerEncoder(DataEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.uint_encoder = UniversalUintEncoder()

    def encode_symbol(self, x: int):
        assert isinstance(x, int)

        #########################
        # ADD CODE HERE
        # Use the self.uint_encoder here
        raise NotImplementedError
        ########################

    def encode_block(self, data_block: DataBlock) -> BitArray:
        encoded_bitarray = BitArray("")
        for s in data_block.data_list:
            encoded_bitarray += self.encode_symbol(s)
        return encoded_bitarray


class UniversalIntegerDecoder(DataDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.uint_decoder = UniversalUintDecoder()

    def decode_symbol(self, encoded_bitarray):
        #########################
        # ADD CODE HERE
        # Use the self.uint_decoder here
        raise NotImplementedError
        ########################

    def decode_block(self, bitarray: BitArray):
        data_list = []
        num_bits_consumed = 0
        while num_bits_consumed < len(bitarray):
            s, num_bits = self.decode_symbol(bitarray[num_bits_consumed:])
            num_bits_consumed += num_bits
            data_list.append(s)

        return DataBlock(data_list), num_bits_consumed


def test_universal_integer_encode_decode():
    """
    The test should pass
    """
    encoder = UniversalIntegerEncoder()
    decoder = UniversalIntegerDecoder()

    # create some sample data
    data_list = [0, 0, -1, 3, -4, 100, -5634]
    data_block = DataBlock(data_list)

    # test encode
    encoded_bitarray = encoder.encode_block(data_block)
    print("Encoded_bitarray length: ", len(encoded_bitarray))

    # test decode
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
    assert num_bits_consumed == len(encoded_bitarray)

    # compare blocks, and check if the encoding is lossless
    assert are_blocks_equal(data_block, decoded_block)
