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
