"""
Contains some elementary baseline compressors
1. Fixed bit width compressor 
"""
import tempfile
import os
from core.data_block import DataBlock
from core.data_encoder_decoder import DataEncoder, DataDecoder
from core.prob_dist import ProbabilityDist
from external_compressors.pickle_external import PickleDecoder, PickleEncoder
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray, get_bit_width
from utils.test_utils import (
    create_random_text_file,
    get_random_data_block,
    try_file_lossless_compression,
    try_lossless_compression,
)
import numpy as np

class AlphabetEncoder(DataEncoder):
    """encode the alphabet set for a block

    Technically, the input to the encode_block is a set and not a DataBlock,
    but we don't plan on using the "encode" function for this class anyway
    """

    def __init__(self):
        self.alphabet_size_bits = 8  # bits used to encode size of the alphabet
        self.alphabet_bits = 8  # bits used to encode each alphabet
        super().__init__()

    def encode_block(self, alphabet):
        """encode the alphabet set

        - Encodes the size of the alphabet set using 8 bits
        - The ascii value corresponding to each alphabet is then encoded using 8 bits
        """
        # encode the alphabet size
        alphabet_size = len(alphabet)
        assert alphabet_size <= 2 ** self.alphabet_size_bits
        alphabet_size_bitarray = uint_to_bitarray(alphabet_size, self.alphabet_size_bits)

        bitarray = alphabet_size_bitarray
        for a in alphabet:
            bitarray += uint_to_bitarray(ord(a), bit_width=self.alphabet_bits)

        return bitarray


class AlphabetDecoder(DataDecoder):
    """decode the encoded alphabet set"""

    def __init__(self):
        self.alphabet_size_bits = 8
        self.alphabet_bits = 8
        super().__init__()

    def decode_block(self, params_data_bitarray: BitArray):
        # initialize num_bits_consumed
        num_bits_consumed = 0

        # get alphabet size
        assert len(params_data_bitarray) >= self.alphabet_size_bits
        alphabet_size = bitarray_to_uint(params_data_bitarray[: self.alphabet_size_bits])
        num_bits_consumed += self.alphabet_size_bits

        alphabet = []
        for _ in range(alphabet_size):
            symbol_bitarray = params_data_bitarray[
                num_bits_consumed : (num_bits_consumed + self.alphabet_bits)
            ]
            symbol = chr(bitarray_to_uint(symbol_bitarray))
            alphabet.append(symbol)
            num_bits_consumed += self.alphabet_bits

        return alphabet, num_bits_consumed


def get_alphabet_fixed_bitwidth(alphabet_size):
    return 1 if (alphabet_size == 1) else int(np.ceil(np.log2(alphabet_size)))


class FixedBitwidthEncoder(DataEncoder):
    """Encode each symbol using a fixed number of bits"""

    def __init__(self, ):
        # serialize the encoder and save the alphabet
        self.alphabet_encoder = PickleEncoder()

    def encode_block(self, data_block: DataBlock):
        """first encode the alphabet and then each data symbol using fixed number of bits"""
        # get bit width
        alphabet = list(data_block.get_alphabet())

        # encode alphabet
        encoded_bitarray = self.alphabet_encoder.encode_block(alphabet)

        # encode data
        symbol_bit_width = get_alphabet_fixed_bitwidth(len(alphabet))
        alphabet_dict = {a: i for i, a in enumerate(alphabet)}
        for s in data_block.data_list:
            encoded_bitarray += uint_to_bitarray(alphabet_dict[s], bit_width=symbol_bit_width)

        return encoded_bitarray


class FixedBitwidthDecoder(DataDecoder):
    def __init__(self):
        # serialize the encoder and save the alphabet
        self.alphabet_decoder = PickleDecoder()

    def decode_block(self, bitarray: BitArray):
        """Decode data encoded by TextFixedBitwidthDecoder

        - retrieve the alphabet
        - the size of the alphabet implies the bit width used for encoding the symbols
        - decode data
        """
        # get the alphabet
        alphabet, num_bits_consumed = self.alphabet_decoder.decode_block(bitarray)

        # decode data
        symbol_bit_width = get_alphabet_fixed_bitwidth(len(alphabet))

        data_list = []
        while num_bits_consumed < len(bitarray):
            symbol_bitarray = bitarray[num_bits_consumed : (num_bits_consumed + symbol_bit_width)]
            ind = bitarray_to_uint(symbol_bitarray)
            data_list.append(alphabet[ind])
            num_bits_consumed += symbol_bit_width

        return DataBlock(data_list), num_bits_consumed

class TextFixedBitwidthEncoder(FixedBitwidthEncoder):
    """Encode each symbol using a fixed number of bits"""
    def __init__(self, ):
        # NOTE: the simple encoder is specific to textual data alphabet
        self.alphabet_encoder = AlphabetEncoder()


class TextFixedBitwidthDecoder(FixedBitwidthDecoder):
    def __init__(self):
        self.alphabet_decoder = AlphabetDecoder()


#########################################


def test_alphabet_encode_decode():
    """test the alphabet compression"""
    # define encoder, decoder
    encoder = AlphabetEncoder()
    decoder = AlphabetDecoder()

    # create some sample data
    alphabet = ["A", "B", "C"]
    output_bits_block = encoder.encode_block(alphabet)
    decoded_alphabet, num_bits_consumed = decoder.decode_block(output_bits_block)
    assert alphabet == decoded_alphabet
    assert num_bits_consumed == (1 + len(alphabet)) * 8


def test_text_fixed_bitwidth_encode_decode():
    """test the encode_block and decode_block functions of FixedBitWidthEncoder and TextFixedBitwidthDecoder"""
    # define encoder, decoder
    encoder = TextFixedBitwidthEncoder()
    decoder = TextFixedBitwidthDecoder()

    # create some sample data
    data_list = ["A", "B", "C", "C", "A", "C", "D"]
    data_block = DataBlock(data_list)

    is_lossless, codelen, _ = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless

    # check if the length of the encoding was correct
    alphabet_bits = (1 + len(data_block.get_alphabet())) * 8
    assert codelen == len(data_list) * 2 + alphabet_bits


def test_text_fixed_bitwidth_file_encode_decode():
    """full test for FixedBitWidthEncoder and TextFixedBitwidthDecoder

    - create a sample file
    - encode the file usnig FixedBitWidthEncoder
    - perform decoding and check if the compression was lossless

    """
    # define encoder, decoder
    encoder = TextFixedBitwidthEncoder()
    decoder = TextFixedBitwidthDecoder()

    with tempfile.TemporaryDirectory() as tmpdirname:

        # create a file with some random data
        input_file_path = os.path.join(tmpdirname, "inp_file.txt")
        create_random_text_file(
            input_file_path,
            file_size=2000,
            prob_dist=ProbabilityDist({"A": 0.5, "B": 0.25, "C": 0.25}),
        )

        # test lossless compression
        assert try_file_lossless_compression(input_file_path, encoder, decoder)


def test_fixed_bitwidth_encode_decode():
    """test the encode_block and decode_block functions of FixedBitWidthEncoder and TextFixedBitwidthDecoder"""
    # define encoder, decoder
    encoder = FixedBitwidthEncoder()
    decoder = FixedBitwidthDecoder()

    # create some sample data
    prob_dist = ProbabilityDist({"A": 0.25, "B": 0.25, "C": 0.2, "D": 0.3})
    DATA_SIZE = 100000
    data_block = get_random_data_block(prob_dist=prob_dist, size=DATA_SIZE, seed=0)

    is_lossless, codelen, _ = try_lossless_compression(data_block, encoder, decoder)
    assert (codelen/DATA_SIZE - 2.0) < 1e-1 #pickle is quite wasteful
    assert is_lossless
