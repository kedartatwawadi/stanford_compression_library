"""
Utility functions useful for testing
"""

import filecmp
from typing import Tuple
from core.data_block import DataBlock
from core.data_stream import TextFileDataStream
from core.data_encoder_decoder import DataDecoder, DataEncoder
from core.prob_dist import ProbabilityDist
from utils.bitarray_utils import BitArray
import tempfile
import os
import numpy as np


def get_random_data_block(prob_dist: ProbabilityDist, size: int, seed: int = None):
    """generates i.i.d random data from the given prob distribution

    Args:
        prob_dist (ProbabilityDist): input probability distribution
        size (int): size of the block to be returned
        seed (int): random seed used to generate the data
    """

    rng = np.random.default_rng(seed)
    data = rng.choice(prob_dist.alphabet, size=size, p=prob_dist.prob_list)
    return DataBlock(data)


def create_random_text_file(file_path: str, file_size: int, prob_dist: ProbabilityDist):
    """creates a random text file at the given path

    Args:
        file_path (str): file path to which random data needs to be written
        file_size (int): The size of the random file to be generated
        prob_dist (ProbabilityDist): the distribution to use to generate the random data
    """
    data_block = get_random_data_block(prob_dist, file_size)
    with TextFileDataStream(file_path, "w") as fds:
        fds.write_block(data_block)


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


def try_lossless_compression(
    data_block: DataBlock, encoder: DataEncoder, decoder: DataDecoder
) -> Tuple[bool, int, BitArray]:
    """
    Encodes the data_block using data_compressor and returns True if the compression was lossless

    Returns:
        Tuple[bool,Int,BitArray]: whether encoding is lossless, size of the output block, encoded bitarray
    """
    # test encode
    output_bits_block = encoder.encode_block(data_block)

    # test decode
    decoded_block, num_bits_consumed = decoder.decode_block(output_bits_block)
    assert num_bits_consumed == len(output_bits_block)

    # compare blocks
    return are_blocks_equal(data_block, decoded_block), len(output_bits_block), output_bits_block


def try_file_lossless_compression(
    input_file_path: str, encoder: DataEncoder, decoder: DataDecoder, encode_block_size=1000
):
    """try encoding the input file and check if it is lossless

    Args:
        input_file_path (str): input file path
        encoder (DataEncoder): encoder object
        decoder (DataDecoder): decoder object
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        encoded_file_path = os.path.join(tmpdirname, "encoded_file.bin")
        reconst_file_path = os.path.join(tmpdirname, "reconst_file.txt")

        # encode data using the FixedBitWidthEncoder and write to the binary file
        encoder.encode_file(input_file_path, encoded_file_path, block_size=encode_block_size)

        # decode data using th eFixedBitWidthDecoder and write output to a text file
        decoder.decode_file(encoded_file_path, reconst_file_path)

        # check if the reconst file and input match
        return filecmp.cmp(input_file_path, reconst_file_path)
