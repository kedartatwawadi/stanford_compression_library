import bitarray
from bitarray.util import ba2int, int2ba
import numpy as np


def get_bit_width(size) -> int:
    return int(np.ceil(np.log2(size)))


# remap bitarray.bitarray for now..
# TODO: we could add more functions later
BitArray = bitarray.bitarray


def uint_to_bitarray(x: int, bit_width=None) -> BitArray:
    """
    converts an unsigned into to bits.
    if bit_width is provided then data is converted accordingly
    """
    return int2ba(x, length=bit_width)


def bitarray_to_uint(bit_array: BitArray) -> int:
    return ba2int(bit_array)


def test_bitarray_to_int():
    """simple tests to verify if uint to bitarray and reverse conversions work"""
    # ex-1
    x = 4
    b = uint_to_bitarray(x)
    assert len(b) == 3
    x_hat = bitarray_to_uint(b)
    assert x == x_hat

    # ex-2
    x = 13
    b = uint_to_bitarray(x, bit_width=8)
    assert len(b) == 8
    x_hat = bitarray_to_uint(b)
    assert x == x_hat
