import bitarray
import numpy as np


def get_bit_width(size) -> int:
    return int(np.ceil(np.log2(size)))


def uint_to_bitstring(uint_data, bit_width=None):
    """
    converts an unsigned into to bits.
    if bit_width is provided then data is converted accordingly
    """
    if bit_width is None:
        return f"{uint_data:b}"
    else:
        return f"{uint_data:0{bit_width}b}"


def bitstring_to_uint(bitstring):
    return int(bitstring, 2)


# remap bitarray.bitarray for now..
# TODO: we could add more functions later
class BitArray(bitarray.bitarray):
    pass


def uint_to_bitarray(x: int, bit_width=None) -> BitArray:
    return BitArray(uint_to_bitstring(x, bit_width=bit_width))


def bitarray_to_uint(bit_array: BitArray) -> int:
    bitstring = bit_array.to01()
    return bitstring_to_uint(bitstring)
