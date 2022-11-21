"""
Utility functions useful for testing
"""

import filecmp
from typing import Tuple
from core.data_block import DataBlock
from core.data_stream import TextFileDataStream, Uint8FileDataStream
from core.data_encoder_decoder import DataDecoder, DataEncoder
from core.prob_dist import Frequencies, ProbabilityDist, get_avg_neg_log_prob
from utils.bitarray_utils import BitArray, get_random_bitarray
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
    return DataBlock(data.tolist())


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


def create_random_binary_file(file_path: str, file_size: int, prob_dist: ProbabilityDist):
    """creates a random binary file at the given path (uses "wb" instead of "w")

    Args:
        file_path (str): file path to which random data needs to be written
        file_size (int): The size of the random file to be generated
        prob_dist (ProbabilityDist): the distribution to use to generate the random data.
                                     The distribution must be on alphabet of bytes/u8's (0-255)
    """
    data_block = get_random_data_block(prob_dist, file_size)
    with Uint8FileDataStream(file_path, "wb") as fds:
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
    data_block: DataBlock,
    encoder: DataEncoder,
    decoder: DataDecoder,
    add_extra_bits_to_encoder_output: bool = False,
    verbose: bool = False
) -> Tuple[bool, int, BitArray]:
    """Encodes the data_block using data_compressor and returns True if the compression was lossless

    Args:
        data_block (DataBlock): input data_block to encode
        encoder (DataEncoder): Encoder obj
        decoder (DataDecoder): Decoder obj to test with
        append_extra_bits_to_encoder_output (bool, optional): This flag adds a random number of slack bits at the end of encoder output.
        This is to test the scenario where we are concatenating multiple encoder outputs in the same bitstream.
        Defaults to False.

    Returns:
        Tuple[bool, int, BitArray]: whether encoding is lossless, size of the output block, encoded bitarray
    """

    # test encode
    encoded_bitarray = encoder.encode_block(data_block)

    # if True, add some random bits to the encoder output
    encoded_bitarray_extra = BitArray(encoded_bitarray)  # make a copy
    if add_extra_bits_to_encoder_output:
        num_extra_bits = int(np.random.randint(100))
        encoded_bitarray_extra += get_random_bitarray(num_extra_bits)

    # test decode
    decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray_extra)
    assert num_bits_consumed == len(encoded_bitarray), "Decoder did not consume all bits"

    # compare blocks
    return are_blocks_equal(data_block, decoded_block), num_bits_consumed, encoded_bitarray


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

        # encode data using the given encoder and write to the binary file
        encoder.encode_file(input_file_path, encoded_file_path, block_size=encode_block_size)

        # decode data using the given decoder and write output to a text file
        decoder.decode_file(encoded_file_path, reconst_file_path)

        # check if the reconst file and input match
        return filecmp.cmp(input_file_path, reconst_file_path)

    """
    """
def lossless_entropy_coder_test(encoder: DataEncoder, decoder: DataDecoder, freq: Frequencies, data_size: int, encoding_optimality_precision: bool = None, seed: int =0):
    """Checks if the given entropy coder performs lossless compression and optionally if it is
       "optimal". 
       
       NOTE: the notion of optimality is w.r.t to the avg_log_probability of the randomly
       generated input.
       Example usage is for compressors such as Huffman, AEC, rANS etc. 

    Args:
        encoder (DataEncoder): Encoder to test with
        decoder (DataDecoder): Decoder to test lossless compression with 
        freq (Frequencies): freq distribution used to generate random i.i.d data
        data_size (int): the size of the data to generate
        encoding_optimality_precision (bool, optional): Optionally (if not None) check if the average log_prob is close to the avg_codelen. Defaults to None.
        seed (int, optional): _description_. seed to generate random data. Defaults to 0.
    """
    # generate random data
    prob_dist = freq.get_prob_dist()
    data_block = get_random_data_block(prob_dist, data_size, seed=seed)
    avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)

    # check if encoding/decoding is lossless
    is_lossless, encode_len, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )

    # avg codelen ignoring the bits used to signal num data elements
    avg_codelen = (encode_len) / data_block.size
    print(
        f" avg_log_prob={avg_log_prob:.3f}, avg_codelen: {avg_codelen:.3f}"
    )

    # check whether arithmetic coding results are close to optimal codelen
    if encoding_optimality_precision is not None:
        err_msg = f"avg_codelen={avg_codelen} is not {encoding_optimality_precision} close to avg_log_prob={avg_log_prob}"
        assert np.abs(avg_codelen - avg_log_prob) < encoding_optimality_precision, err_msg

    assert is_lossless

def lossless_test_against_expected_bitrate(
    encoder: DataEncoder,
    decoder: DataDecoder,
    data_block: DataBlock,
    expected_bitrate: float,
    encoding_optimality_precision: float,
):
    """Checks encoder/decoder for losslessness and also against expected bitrate.

    Args:
        encoder (DataEncoder): Encoder to test with
        decoder (DataDecoder): Decoder to test lossless compression with
        data_block (DataBlock): data to use for testing
        expected_bitrate (float): the theoretically expected bitrate
        encoding_optimality_precision (float): check that the average expected_bitrate is close to the avg_codelen
    """
    # check if encoding/decoding is lossless
    is_lossless, encode_len, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )

    # avg codelen ignoring the bits used to signal num data elements
    avg_codelen = (encode_len) / data_block.size
    print(f" expected_bitrate={expected_bitrate:.3f}, avg_codelen: {avg_codelen:.3f}")

    # check whether arithmetic coding results are close to expected codelen
    err_msg = f"avg_codelen={avg_codelen} is not {encoding_optimality_precision} close to expected_bitrate={expected_bitrate}"
    assert np.abs(avg_codelen - expected_bitrate) < encoding_optimality_precision, err_msg

    assert is_lossless
