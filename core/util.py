from typing import Set, List


def compute_alphabet(data_list: List) -> Set:
    alphabet = set()
    for d in data_list:
        alphabet.add(d)
    return alphabet


def compute_counts_dict(data_list: List) -> dict:
    """
    returns a dict of the counts of each symbol in the data_list
    """
    # get the alphabet
    alphabet = compute_alphabet(data_list)

    # initialize the count dict
    count_dict = {}
    for a in alphabet:
        count_dict[a] = 0

    # populate the count dict
    for d in data_list:
        count_dict[d] += 1

    return count_dict


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
