from typing import Tuple
from core.data_encoder_decoder import DataDecoder, DataEncoder
from core.data_block import DataBlock
import argparse
from core.prob_dist import ProbabilityDist
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder, HuffmanTree
from core.data_stream import Uint8FileDataStream
from core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
import pickle

# constants
BLOCKSIZE = 50_000  # encode in 50 KB blocks

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--decompress", help="decompress", action="store_true")
parser.add_argument("-i", "--input", help="input file", required=True, type=str)
parser.add_argument("-o", "--output", help="output file", required=True, type=str)


def encode_prob_dist(prob_dist: ProbabilityDist) -> BitArray:
    """Encode a probability distribution as a bit array

    Args:
        prob_dist (ProbabilityDist): probability distribution over 0, 1, 2, ..., 255
            (note that some probabilities might be missing if they are 0).

    Returns:
        BitArray: encoded bit array
    """
    #########################
    # ADD CODE HERE
    # bits = BitArray(), bits.frombytes(byte_array), uint_to_bitarray might be useful to implement this
    raise NotImplementedError("You need to implement encode_prob_dist")
    #########################

    return encoded_probdist_bitarray


def decode_prob_dist(bitarray: BitArray) -> Tuple[ProbabilityDist, int]:
    """Decode a probability distribution from a bit array

    Args:
        bitarray (BitArray): bitarray encoding probability dist followed by arbitrary data

    Returns:
        prob_dit (ProbabilityDist): the decoded probability distribution
        num_bits_read (int): the number of bits read from bitarray to decode probability distribution
    """
    #########################
    # ADD CODE HERE
    # bitarray.tobytes() and bitarray_to_uint() might be useful to implement this
    raise NotImplementedError("You need to implement decode_prob_dist")
    #########################

    return prob_dist, num_bits_read


def print_huffman_tree(prob_dist):
    """Print Huffman tree after changing to ASCII symbols"""
    prob_dist = ProbabilityDist({chr(k): prob_dist.prob_dict[k] for k in prob_dist.prob_dict})
    tree = HuffmanTree(prob_dist)
    tree.print_tree()


class HuffmanEmpiricalEncoder(DataEncoder):
    def encode_block(self, data_block: DataBlock):

        # get the empirical distribution of the data block
        prob_dist = data_block.get_empirical_distribution()

        # uncomment below to print Huffman tree
        # print_huffman_tree(prob_dist)

        # create Huffman encoder for the empirical distribution
        huffman_encoder = HuffmanEncoder(prob_dist)
        # encode the data with Huffman code
        encoded_data = huffman_encoder.encode_block(data_block)
        # return the Huffman encoding prepended with the encoded probability distribution
        return encode_prob_dist(prob_dist) + encoded_data

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int):
        """utility wrapper around the encode function using Uint8FileDataStream
        Args:
            input_file_path (str): path of the input file
            encoded_file_path (str): path of the encoded binary file
            block_size (int): choose the block size to be used to call the encode function
        """
        # call the encode function and write to the binary file
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class HuffmanEmpiricalDecoder(DataDecoder):
    def decode_block(self, encoded_block: DataBlock):
        # first decode the probability distribution
        prob_dist, num_bits_read_prob_dist_encoder = decode_prob_dist(encoded_block)
        # now create Huffman decoder
        huffman_decoder = HuffmanDecoder(prob_dist)
        # now apply Huffman decoding
        decoded_data, num_bits_read_huffman = huffman_decoder.decode_block(
            encoded_block[num_bits_read_prob_dist_encoder:]
        )
        # verify we read all the bits provided
        assert num_bits_read_huffman + num_bits_read_prob_dist_encoder == len(encoded_block)
        return decoded_data, len(encoded_block)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function using Uint8FileDataStream
        Args:
            encoded_file_path (str): input binary file
            output_file_path (str): output (text) file to which decoded data is written
        """
        # read from a binary file and decode data and write to a binary file
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


def test_encode_decode_prob_dist():
    prob_dist = ProbabilityDist({0: 0.44156346, 1: 0.23534656, 255: 0.32308998})
    encoded_bits = encode_prob_dist(prob_dist)
    decoded_prob_dist, num_bits_read = decode_prob_dist(encoded_bits)
    assert decoded_prob_dist.prob_dict == prob_dist.prob_dict, "decoded prob dist does not match original"
    assert num_bits_read == len(encoded_bits), "All encoded bits were not consumed by the decoder"


if __name__ == "__main__":
    args = parser.parse_args()
    if args.decompress:
        decoder = HuffmanEmpiricalDecoder()
        decoder.decode_file(args.input, args.output)
    else:
        encoder = HuffmanEmpiricalEncoder()
        encoder.encode_file(args.input, args.output, block_size=BLOCKSIZE)
