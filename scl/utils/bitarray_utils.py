import bitarray
from bitarray.util import ba2int, int2ba
import numpy as np
from operator import index
from typing import Tuple
from typing_extensions import TypeAlias


def get_bit_width(x) -> int:
    """get the minimum number of bits needed to represent the input uint x

    Args:
        x (int): input unsigned integer

    Returns:
        int:
    """
    x = index(x)
    if x < 0:
        raise ValueError("x must be a nonnegative integer")
    return max(1, x.bit_length())


# remap bitarray.bitarray for now..
# TODO: we could add more functions later
BitArray: TypeAlias = bitarray.bitarray


def uint_to_bitarray(x: int, bit_width=None) -> BitArray:
    """
    converts an unsigned int to bits.
    if bit_width is provided then data is converted accordingly
    """
    assert isinstance(x, (int, np.integer))
    return int2ba(int(x), length=bit_width)  # int2ba requires input to be dtype int


def bitarray_to_uint(bit_array: BitArray) -> int:
    return ba2int(bit_array)


def get_random_bitarray(size) -> BitArray:
    return bitarray.util.urandom(size)


def float_to_bitarrays(x: float, max_precision: int) -> Tuple[BitArray, BitArray]:
    """convert floating point number to binary with the given max_precision

    Utility function to obtain binary representation of the floating point number.
    We return a tuple of binary representations of the integer part and the fraction part of the
    floating point number

    Args:
        x (float): inpout floating point number
        max_precision (int): max binary precision (after the decimal point) to which we should return the bitarray
    Returns:
        Tuple[BitArray, BitArray]: returns (uint_x_bitarray, frac_x_bitarray)
    """

    # find integer, fraction part of x
    uint_x = int(x)
    frac_x = x - int(x)

    # obtain binary representations of integer and fractional parts
    int_x_bitarray = uint_to_bitarray(uint_x)
    frac_x_bitarray = uint_to_bitarray(
        int(frac_x * np.power(2, max_precision)), bit_width=max_precision
    )
    return int_x_bitarray, frac_x_bitarray


def bitarrays_to_float(uint_x_bitarray: BitArray, frac_x_bitarray: BitArray) -> float:
    """converts bitarrays corresponding to integer and fractional part of a floatating point number to a float

    Args:
        uint_x_bitarray (BitArray): bitarray corresponding to the integer part of x
        frac_x_bitarray (BitArray): bitarray corresponding to the fractional part of x

    Returns:
        float: x, the floating point number
    """
    # convert uint_x_bitarray to the integer part of the float
    uint_x = bitarray_to_uint(uint_x_bitarray)

    # convert frac_x_bitarray to the fractional part of the float
    precision = len(frac_x_bitarray)
    frac_x = bitarray_to_uint(frac_x_bitarray) / (np.power(2, precision))

    return uint_x + frac_x


############################## TESTS ####################################


def test_basic_bitarray_operations():
    # testing if iterating through bitarray works
    # and if/else condition work on a bit
    code = BitArray("01011")
    for bit in code:
        if bit:
            assert bit == 1
        else:
            assert bit == 0


def test_get_bit_width():
    """check if get_bit_width returns the correct value for different inputs"""
    assert get_bit_width(0) == 1
    assert get_bit_width(1) == 1
    assert get_bit_width(1 << 16) == 17
    assert get_bit_width(255) == 8


def test_get_bit_width_exact_boundaries():
    for exponent in [1, 2, 31, 32, 53, 63, 64, 127, 1024]:
        assert get_bit_width((1 << exponent) - 1) == exponent
        assert get_bit_width(1 << exponent) == exponent + 1
        assert get_bit_width((1 << exponent) + 1) == exponent + 1
    assert get_bit_width(np.uint8(255)) == 8
    assert get_bit_width(np.int64(2**63 - 1)) == 63
    assert get_bit_width(np.uint64(2**64 - 1)) == 64


def test_get_bit_width_rejects_invalid_inputs():
    for value in [-1, np.int64(-1)]:
        with np.testing.assert_raises(ValueError):
            get_bit_width(value)
    for value in [0.0, 1.5, np.float64(2), "1"]:
        with np.testing.assert_raises(TypeError):
            get_bit_width(value)


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


def test_float_to_bitarrays():
    """simple tests to verify if float to bitarray and reverse works"""
    # ex-1
    x = 0.25
    max_precision = 2
    uint_x_bitarray, frac_x_bitarray = float_to_bitarrays(x, max_precision=max_precision)

    assert len(frac_x_bitarray) == max_precision
    x_hat = bitarrays_to_float(uint_x_bitarray, frac_x_bitarray)
    np.testing.assert_almost_equal(x, x_hat)
