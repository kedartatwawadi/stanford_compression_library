"""
A simple compressor which serializes any given input data structure
using pickle and converts it to bitarray. 

This compresor is useful when we want to send over some metadata/side-information
in the bitstream whose size is not a big concern

NOTE: pickle is extremenly wasteful, and we can do much better with custom serialization
so this should be only used when the metadata is small.
"""

import pickle
from typing import Any
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray


class PickleEncoder(DataEncoder):
    def __init__(self, length_bitwidth=32):
        self.length_bitwidth = length_bitwidth

    def encode_block(self, data: Any):
        # pickle prob dist and convert to bytes
        pickled_bits = BitArray()
        bytes = pickle.dumps(data)
        pickled_bits.frombytes(bytes)
        len_pickled = len(pickled_bits)

        # encode length of pickled data
        assert len_pickled < (1 << self.length_bitwidth)
        length_encoding = uint_to_bitarray(len_pickled, bit_width=self.length_bitwidth)
        return length_encoding + pickled_bits


class PickleDecoder(DataDecoder):
    def __init__(self, length_bitwidth=32):
        self.length_bitwidth = length_bitwidth

    def decode_block(self, bitarray: BitArray):
        length_encoding = bitarray[:self.length_bitwidth]
        len_pickled = bitarray_to_uint(length_encoding)
        # bits to bytes
        pickled_bytes = bitarray[self.length_bitwidth : self.length_bitwidth + len_pickled].tobytes()

        decoded_data = pickle.loads(pickled_bytes)
        num_bits_read = self.length_bitwidth + len_pickled
        return decoded_data, num_bits_read


def test_pickle_data_compressor():
    p_enc = PickleEncoder()
    p_dec = PickleDecoder()

    # pickle should work for arbitrary data
    data_list = [3, "alpha", "33.2313241234"]
    encoded_bits = p_enc.encode_block(data_list)
    encoded_bits_and_extra_bits = encoded_bits + BitArray("101111")
    data_list_decoded, num_bits_consumed = p_dec.decode_block(encoded_bits_and_extra_bits)
    assert num_bits_consumed == len(encoded_bits)
    for d1, d2 in zip(data_list, data_list_decoded):
        assert d1 == d2

    data_ordered_dict = {"A": 1.111, "B": 0.3412452, "C": 0.1213441}
    encoded_bits = p_enc.encode_block(data_ordered_dict)
    encoded_bits_and_extra_bits = encoded_bits + BitArray("101111")
    data_dict_decoded, num_bits_consumed = p_dec.decode_block(encoded_bits_and_extra_bits)
    assert num_bits_consumed == len(encoded_bits)

    for d1, d2 in zip(data_ordered_dict, data_dict_decoded):
        assert d1 == d2
        assert data_ordered_dict[d1] == data_dict_decoded[d2]
