"""
Contains some elementary baseline compressors
1. Fixed bit width compressor 
"""

from core.data_block import DataBlock
from core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
from core.framework import DataEncoder, DataDecoder
from core.util import BitArray, bitarray_to_uint, uint_to_bitarray, get_bit_width
from utils.test_utils import try_lossless_compression
import tempfile
import os
from core.data_stream import TextFileDataStream
import filecmp


class AlphabetEncoder(DataEncoder):
    def __init__(self):
        self.alphabet_size_bits = 8
        super().__init__()

    def encode_block(self, alphabet):
        # encode the alphabet size
        alphabet_size = len(alphabet)
        assert alphabet_size < 2 ** self.alphabet_size_bits
        alphabet_size_bitarray = uint_to_bitarray(alphabet_size, self.alphabet_size_bits)

        bitarray = alphabet_size_bitarray
        for a in alphabet:
            bitarray += uint_to_bitarray(ord(a), bit_width=self.alphabet_size_bits)

        return bitarray


class AlphabetDecoder(DataDecoder):
    def __init__(self):
        self.alphabet_size_bits = 8
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
                num_bits_consumed : (num_bits_consumed + self.alphabet_size_bits)
            ]
            symbol = chr(bitarray_to_uint(symbol_bitarray))
            alphabet.append(symbol)
            num_bits_consumed += self.alphabet_size_bits

        return alphabet, num_bits_consumed


class FixedBitwidthEncoder(DataEncoder):
    def __init__(self):
        super().__init__()
        self.alphabet_encoder = AlphabetEncoder()

    def encode_block(self, data_block: DataBlock):
        # get bit width
        alphabet = data_block.get_alphabet()

        # encode alphabet
        encoded_bitarray = self.alphabet_encoder.encode_block(alphabet)

        # encode data
        symbol_bit_width = get_bit_width(len(alphabet))
        alphabet_dict = {a: i for i, a in enumerate(alphabet)}
        for s in data_block.data_list:
            encoded_bitarray += uint_to_bitarray(alphabet_dict[s], bit_width=symbol_bit_width)

        return encoded_bitarray


class FixedBitwidthDecoder(DataDecoder):
    def __init__(self):
        super().__init__()
        self.alphabet_decoder = AlphabetDecoder()

    def decode_block(self, bitarray: BitArray):
        # get the alphabet
        alphabet, num_bits_consumed = self.alphabet_decoder.decode_block(bitarray)

        # decode data
        symbol_bit_width = get_bit_width(len(alphabet))

        data_list = []
        while num_bits_consumed < len(bitarray):
            symbol_bitarray = bitarray[num_bits_consumed : (num_bits_consumed + symbol_bit_width)]
            ind = bitarray_to_uint(symbol_bitarray)
            data_list.append(alphabet[ind])
            num_bits_consumed += symbol_bit_width

        return DataBlock(data_list), num_bits_consumed


def test_alphabet_encode_decode():
    # define encoder, decoder
    encoder = AlphabetEncoder()
    decoder = AlphabetDecoder()

    # create some sample data
    alphabet = ["A", "B", "C"]
    output_bits_block = encoder.encode_block(alphabet)
    decoded_alphabet, num_bits_consumed = decoder.decode_block(output_bits_block)
    assert alphabet == decoded_alphabet
    assert num_bits_consumed == (1 + len(alphabet)) * 8


def test_fixed_bitwidth_encode_decode():
    # define encoder, decoder
    encoder = FixedBitwidthEncoder()
    decoder = FixedBitwidthDecoder()

    # create some sample data
    data_list = ["A", "B", "C", "C", "A", "C"]
    data_block = DataBlock(data_list)

    is_lossless, codelen = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless

    # check if the length of the encoding was correct
    alphabet_bits = (1 + len(data_block.get_alphabet())) * 8
    assert codelen == len(data_list) * 2 + alphabet_bits


def test_fixed_bitwidth_file_write():
    # define encoder, decoder
    encoder = FixedBitwidthEncoder()
    decoder = FixedBitwidthDecoder()

    # write data to file
    # create a temporary file
    with tempfile.TemporaryDirectory() as tmpdirname:
        input_file_path = os.path.join(tmpdirname, "inp_file.txt")
        encoded_file_path = os.path.join(tmpdirname, "encoded_file.bin")
        reconst_file_path = os.path.join(tmpdirname, "reconst_file.txt")

        # create some sample data
        data_gt = [DataBlock(["AB"] * 1000), DataBlock(["CDE"] * 500)]

        # write data to the file
        with TextFileDataStream(input_file_path, "w") as fds:
            fds.write_block(data_gt[0])
            fds.write_block(data_gt[1])

        # read data from the file
        with TextFileDataStream(input_file_path, "r") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                encoder.encode(fds, block_size=500, encode_writer=writer)

        # decode data from the file
        with TextFileDataStream(reconst_file_path, "w") as fds:
            with EncodedBlockReader(encoded_file_path) as reader:
                decoder.decode(reader, fds)

        assert filecmp(input_file_path, reconst_file_path)
