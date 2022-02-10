import abc
from core.util import bitstring_to_uint, uint_to_bitstring
import numpy as np
import bitarray

# data block
# bits_block + pad

# remap bitarray.bitarray for now..
# TODO: we could add more functions later
BitArray = bitarray.bitarray


def uint_to_bitarray(x: int, bit_width=None) -> BitArray:
    assert x >= 0
    if bit_width is None:
        return BitArray(x)
    ret = BitArray(x)
    pad = BitArray("0") * (bit_width - len(ret))
    return pad + ret


def bitarray_to_uint(bit_array: BitArray) -> int:
    return int(bit_array.to01(), 2)


#################


class Padder:
    NUM_PAD_BITS = 3

    @classmethod
    def add_byte_padding(cls, payload_bitarray: BitArray) -> BitArray:
        assert isinstance(payload_bitarray, BitArray)
        payload_size = len(payload_bitarray)
        num_pad = (8 - (payload_size + cls.NUM_PAD_BITS) % 8) % 8

        padding = uint_to_bitstring(num_pad, bit_width=cls.NUM_PAD_BITS) + "0" * num_pad
        padding_bitarray = BitArray(padding)
        return padding_bitarray + payload_bitarray

    @classmethod
    def remove_byte_padding(cls, payload_pad_bitarray: BitArray) -> BitArray:
        assert isinstance(payload_pad_bitarray, BitArray)
        # get padding
        pad_bitarray = payload_pad_bitarray[: cls.NUM_PAD_BITS]
        num_pad = bitarray_to_uint(pad_bitarray)

        # header
        payload_bitarray = payload_pad_bitarray[cls.NUM_PAD_BITS + num_pad :]
        return payload_bitarray


def test_padder():
    def _test(bits_gt):
        # add padding
        padded_bits_gt = Padder.add_byte_padding(bits_gt)
        assert len(padded_bits_gt) % 8 == 0

        # remove padding
        padding_removed_bits = Padder.remove_byte_padding(padded_bits_gt)
        assert bits_gt == padding_removed_bits

    payloads = [BitArray("10110"), BitArray("1" * 23)]
    for payload in payloads:
        _test(payload)


#################


class HeaderHandler:
    NUM_HEADER_BYTES = 4


def bits_to_bytes(bits_list):
    assert len(bits_list) % 8 == 0
    return np.packbits(bits_list)


def bytes_to_bits(bytes_list):
    return np.unpackbits(bytes_list)


def test_bits_bytes_conversion():
    payload = np.array([23, 78], dtype=np.uint8)

    # bytes to bits
    bits = bytes_to_bits(payload)
    assert len(bits) == len(payload) * 8

    # bits to bytes
    bytes_arr = bits_to_bytes(bits)
    np.testing.assert_array_equal(bytes_arr, payload)


################


def add_header(bytes_arr):
    arr_size = len(bytes_arr)
    size_bytes = list(np.frombuffer(arr_size.to_bytes(4, "big"), dtype=np.uint8))
    return size_bytes + bytes_arr


def test_header():
    payload = np.array([23, 78], dtype=np.uint8)
    bytes = add_header(payload)


# def get_block_header(payload_size: int):
#     assert isinstance(payload_size, int)

#     # add a limit on payload_size, which helps simplify the header decoding
#     assert payload_size <= MAX_PAYLOAD_SIZE
#     bitstring = uint_to_bitstring(payload_size)
#     len_bitstring = len(bitstring) * "0" + "1"
#     header = (len_bitstring + bitstring).split()
#     return header

# def get_byte_padding(payload_header_size: int):
#     num_pad = (payload_header_size + NUM_PAD_BITS)%8
#     padding = uint_to_bitstring(num_pad, bit_width=NUM_PAD_BITS) + "0"*num_pad
#     padding_bits = uint_to_bitstring(padding)
#     return padding_bits

# def add_header_and_padding(payload_bits_list):
#     header = get_block_header(len(payload_bits_list))
#     header_and_payload = header + payload_bits_list
#     padding = get_byte_padding(len(header_and_payload))
#     return (padding + header_and_payload)

# def get_bytes_from_bitlist(bits_list):
#     assert len(bits_list)%8 == 0
#     return np.packbits(bits_list)
#     #return bytes([int("".join(map(str, bits_list[i:i+8])), 2) for i in range(0, len(bits_list), 8)])


# def get_payload_and_header_size(bytes_list):
#     bits_list = np.unpackbits(bytes_list)

#     # get padding
#     pad_bitstring = ''.join(bits_list[:NUM_PAD_BITS])
#     num_pad = bitstring_to_uint(pad_bitstring)

#     # header
#     header_bits = bits_list[NUM_PAD_BITS+num_pad:]


# def decoder_bits_parser(data_block, start_ind):

#     # infer the length
#     num_ones = 0
#     for ind in range(start_ind, data_block.size):
#         bit = data_block.data_list[ind]
#         if str(bit) == "0":
#             break
#         num_ones += 1

#     # compute the new start_ind
#     new_start_ind = 2 * num_ones + 1 + start_ind

#     # decode the symbol
#     bitstring = "".join(data_block.data_list[start_ind + num_ones + 1 : new_start_ind])
#     symbol = bitstring_to_uint(bitstring)

#     return symbol, new_start_ind
# class EncodeFileWriter:
#     def __init__(self, file_path: str):
#         self.file_path = file_path

#     def __enter__(self):
#         self.file_reader = open(self.file_path, "wb")
#         return self

#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         self.file_reader.close()

#     def generate_header(self, data_size):


#     @abc.abstractmethod
#     def reset(self):
#         # resets the data stream
#         pass

#     @abc.abstractmethod
#     def get_next_symbol(self):
#         pass # returns None if the stream is finished

#     def get_next_data_block(self, block_size: int):
#         # returns the next data block
#         data_list = []
#         for _ in range(block_size):
#             # get next symbol
#             s = self.get_next_symbol()
#             if s is None:
#                 break
#             data_list.append(s)

#         # if data_list is empty, return None to signal the stream is over
#         if not data_list:
#             return None

#         return DataBlock(data_list)
