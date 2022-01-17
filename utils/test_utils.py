"""
Utility functions useful for testing
"""

from typing import Tuple
from core.data_block import DataBlock
from core.data_compressor import DataCompressor


def are_blocks_equal(data_block_1: DataBlock, data_block_2: DataBlock):
    """
    return True is the blocks are equal
    """
    if data_block_1.size != data_block_2.size:
        return False

    # check if the encoding/decoding was lossless
    for inp_symbol, out_symbol in zip(data_block_1.data_list, data_block_2.data_list):
        if inp_symbol != out_symbol:
            return False

    return True


def try_lossless_compression(data_block: DataBlock, compressor: DataCompressor) -> Tuple[bool, int]:
    """
    Encodes the data_block using data_compressor and returns True if the compression was lossless
    returns (True/False, size of the output block)
    """
    try:
        # test encode
        output_bits_block = compressor.encode(data_block)

        # test decode
        decoded_block = compressor.decode(output_bits_block)

        # compare blocks
        return are_blocks_equal(data_block, decoded_block), output_bits_block.size

    except:
        print("Error during Encoding/Decoding")
        return False, 0
