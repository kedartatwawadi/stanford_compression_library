from typing import Tuple
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_block import DataBlock
import argparse
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
from scl.compressors.huffman_coder import HuffmanEncoder, HuffmanDecoder, HuffmanTree
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
import struct

# constants
BLOCKSIZE = 50_000  # encode in 50 KB blocks

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--decompress", help="decompress", action="store_true")
parser.add_argument("-i", "--input", help="input file", required=True, type=str)
parser.add_argument("-o", "--output", help="output file", required=True, type=str)

def int_to_bitarray8(x: int) -> BitArray:
    """Converts an integer from 0..255 to a bitarray of length 8
    Args:
        x (int): input integer in 0..255
    Returns:
        BitArray: bitarray of length 8 representing the integer
    """
    assert 0 <= x <= 255
    return uint_to_bitarray(x, bit_width=8)

def float_to_bitarray64(x: float) -> BitArray:
    """Converts a Python float to a bitarray of length 64
    Args:
        x (float): input floating point number
    Returns:
        BitArray: bitarray of length 64 representing the float
    """
    float_bytes = struct.pack('d', x)
    float_bits = BitArray()
    float_bits.frombytes(float_bytes)
    assert len(float_bits) == 64
    return float_bits

def bitarray8_to_int(bitarray: BitArray) -> int:
    """Converts a bitarray of length 8 to an integer in 0..255
    Args:
        bitarray (BitArray): input bitarray of length 8
    Returns:
        int: the integer represented by the bitarray
    """
    assert len(bitarray) == 8
    return bitarray_to_uint(bitarray)

def bitarray64_to_float(bitarray: BitArray) -> float:
    """Converts a bitarray of length 64 to a Python float
    Args:
        bitarray (BitArray): input bitarray of length 64
    Returns:
        float: the floating point number represented by the bitarray
    """
    assert len(bitarray) == 64
    float_bytes = bitarray.tobytes()
    x = struct.unpack('d', float_bytes)[0]
    return x

def encode_prob_dist(prob_dist: ProbabilityDist) -> BitArray:
    """Encode a probability distribution as a bit array

    Args:
        prob_dist (ProbabilityDist): probability distribution over 0, 1, 2, ..., 255
            (note that some probabilities might be missing if they are 0).

    Returns:
        BitArray: encoded bit array
    """
    prob_dict = prob_dist.prob_dict # dictionary mapping symbols to probabilities

    #########################
    # ADD CODE HERE
    # You can find int_to_bitarray8 and float_to_bitarray64 useful
    # to encode the symbols and the probabilities respectively.
    # uint_to_bitarray from utils.bitarray_utils can also come in handy
    # to encode any other integer values your solution requires.

    raise NotImplementedError("You need to implement encode_prob_dist")
    #########################

    return encoded_probdist_bitarray


def decode_prob_dist(bitarray: BitArray) -> Tuple[ProbabilityDist, int]:
    """Decode a probability distribution from a bit array

    Args:
        bitarray (BitArray): bitarray encoding probability dist followed by arbitrary data

    Returns:
        prob_dist (ProbabilityDist): the decoded probability distribution
        num_bits_read (int): the number of bits read from bitarray to decode probability distribution
    """
    #########################
    # ADD CODE HERE
    # You can find bitarray8_to_int and bitarray64_to_float useful
    # to decode the symbols and the probabilities respectively.
    # bitarray_to_uint from utils.bitarray_utils can also come in handy
    # to decode any other integer values your solution requires.

    raise NotImplementedError("You need to implement decode_prob_dist")
    #########################

    prob_dist = ProbabilityDist(prob_dict)
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


"""
Helper function to verify roundtrip encoding and decoding of probability distribution.
"""
def helper_encode_decode_prob_dist_roundtrip(prob_dist: ProbabilityDist):
    encoded_bits = encode_prob_dist(prob_dist)
    len_encoded_bits = len(encoded_bits)
    assert type(encoded_bits) == BitArray, "Type of encoded_bits is not BitArray"
    # add some garbage bits in the end to make sure they are not consumed by the decoder
    encoded_bits += BitArray("1010")
    decoded_prob_dist, num_bits_read = decode_prob_dist(encoded_bits)
    assert decoded_prob_dist.prob_dict == prob_dist.prob_dict, "decoded prob dist does not match original"
    assert list(decoded_prob_dist.prob_dict.keys()) == list(prob_dist.prob_dict.keys()), "order of symbols changed"
    assert num_bits_read == len_encoded_bits, "All encoded bits were not consumed by the decoder"

def test_encode_decode_prob_dist():
    # try a simple probability distribution
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({0: 0.4, 1: 0.2, 255: 0.4}))

    # try an uglier looking distribution
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({0: 0.44156346, 1: 0.23534656, 255: 0.32308998}))

    # reorder the symbols to make sure the implementation preserves the order
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({0: 0.4, 255: 0.4, 1: 0.2}))

    # an example with all symbols in 0..255
    helper_encode_decode_prob_dist_roundtrip(ProbabilityDist({i: 1/256 for i in range(256)}))

    # now get a real distribution based on this file
    # load current file as a byte array and get the empirical distribution
    with open(__file__, "rb") as f:
        file_bytes = f.read()
    prob_dist = DataBlock(file_bytes).get_empirical_distribution()
    helper_encode_decode_prob_dist_roundtrip(prob_dist)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.decompress:
        decoder = HuffmanEmpiricalDecoder()
        decoder.decode_file(args.input, args.output)
    else:
        encoder = HuffmanEmpiricalEncoder()
        encoder.encode_file(args.input, args.output, block_size=BLOCKSIZE)
