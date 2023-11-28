import random
from scl.utils.multiset_utils import MultiSetNode
from math import log2, factorial

def rans_encode(state: int, cumul_count: int, incidence: int, upper_bound: int):
    # Standard rANS encode
    block_id = state // incidence
    slot = cumul_count + (state % incidence)
    next_state = block_id * upper_bound + slot
    return next_state

def rans_decode_slot(state: int, upper_bound: int):
    # Return just the slot (the actual symbol decode happens
    # via multiset)
    slot = state % upper_bound
    return slot

def rans_decode(state: int, cumul_count: int, incidence: int, upper_bound: int):
    block_id = state // upper_bound
    slot = state % upper_bound

    # Retrieve previous state
    prev_state = (
        block_id * incidence
        + slot
        - cumul_count
    )
    return prev_state

def swor_encode(state: int, symbol: int, multiset: MultiSetNode):
    # Add symbol to multiset under construction
    multiset.insert(symbol)

    # Lookup symbol distbn info in multiset
    cumul_count, incidence = multiset.forward_lookup(symbol)

    # Encode via rANS given distbn info
    state = rans_encode(state, cumul_count, incidence, multiset.size)
    return state

def swor_decode(state: int, multiset: MultiSetNode):
    multiset_size = multiset.size

    # Decode slot (cumulative count)
    slot = rans_decode_slot(state, multiset_size)

    # Lookup symbol by slot and remove from multiset
    symbol, (cumul_count, incidence) = multiset.reverse_lookup(slot)
    multiset.remove(symbol)

    # Run full rANS decode to output new state, given distbn info
    state = rans_decode(state, cumul_count, incidence, multiset_size)
    return state, symbol

def multiset_encode(state: int, multiset: MultiSetNode, symbol_encode):
    while not multiset.empty:
        # Sample symbol
        state, symbol = swor_decode(state, multiset)

        # Encode symbol
        state = symbol_encode(state, symbol)
    return state

def multiset_decode(state: int, multiset_size: int, symbol_decode):
    multiset = MultiSetNode()
    for _ in range(multiset_size):
        # Decode symbol
        state, symbol = symbol_decode(state)

        # Encode back bits for sampling
        state = swor_encode(state, symbol, multiset)
    return state, multiset


######################################## TESTS ##########################################

def generate_e2e_test(multiset_size: int, alphabet_size: int = 5):
    random.seed(42)
    per_char = multiset_size // alphabet_size
    assert per_char > 0
    chars = sum([[chr(x)] * per_char for x in range(ord("A"), ord("A") + alphabet_size)], [])
    # chars = ["a", "a", "b"]

    orig_multiset = MultiSetNode.from_iterable(chars)
    multiset = orig_multiset.clone()
    print("Multiset size:", multiset_size)

    def symbol_encode(state, symbol):
        # Uniform prior
        return rans_encode(state, ord(symbol), 1, 128)

    def symbol_decode(state):
        # Uniform prior
        symbol = rans_decode_slot(state, 128)
        state = rans_decode(state, symbol, 1, 128)
        return state, symbol

    initial_state = 2023
    encoded_state = multiset_encode(initial_state, multiset, symbol_encode)
    print("Encoded state:", encoded_state)

    output_state, decoded_multiset = multiset_decode(encoded_state, multiset_size, symbol_decode)
    assert initial_state == output_state

    decoded_multiset.map_values(chr)
    print("Decoded multiset:", decoded_multiset)
    assert orig_multiset == decoded_multiset

    char_state = initial_state
    for char in chars:
        char_state = symbol_encode(char_state, char)
    print("Char state:", char_state)
    encoded_char_state = char_state

    decoded_chars = []
    for _ in range(len(chars)):
        char_state, char = symbol_decode(char_state)
        decoded_chars.append(chr(char))
    assert decoded_chars[::-1] == chars

    print("Multiset bits used:", log2(encoded_state))
    print("Naive bits used:", log2(encoded_char_state))
    print("Improvement (bits):", log2(encoded_char_state) - log2(encoded_state))
    print(f"Improvement: {(log2(encoded_char_state) - log2(encoded_state)) / log2(encoded_char_state):.1%}")
    print("Theoretical Improvement (bits):", log2(factorial(multiset_size)))
    print(f"% Achieved of Theoretical Improvement: {(log2(encoded_char_state) - log2(encoded_state)) / log2(factorial(multiset_size)):.1%}")
    return {
        "multiset_bits_used": log2(encoded_state),
        "naive_bits_used": log2(encoded_char_state),
        "theoretical_bits_used": log2(factorial(multiset_size)),
        "improvement": (log2(encoded_char_state) - log2(encoded_state)) / log2(encoded_char_state),
        "theoretical_improvement": (log2(encoded_char_state) - log2(encoded_state)) / log2(factorial(multiset_size)),
        "improvement_bits": log2(encoded_char_state) - log2(encoded_state),
        "theoretical_improvement_bits": log2(factorial(multiset_size)) - log2(encoded_state),
    }


def test_e2e_freq_map():
    # Using matplotlib, plot the % improvement in bits as a function of the
    # multiset size in the following loop
    import matplotlib.pyplot as plt

    multiset_sizes = [50, 100, 200, 500, 1000]

    # plot more curves for alphabet sizes 3, 6, 12, 18, 22, and 26 with different colors
    for alphabet_size in [3, 6, 12, 18, 22, 26]:
        # when choosing multiset_size in multiset_sizes, we need to subtract the remainder to make sure the alphabet_size evenly divides the multiset_size
        adjusted_multiset_sizes = [multiset_size - (multiset_size % alphabet_size) for multiset_size in multiset_sizes]

        results = [
            generate_e2e_test(multiset_size, alphabet_size)["improvement"] * 100
            for multiset_size in adjusted_multiset_sizes
        ]
        plt.plot(adjusted_multiset_sizes, results, label=f"Alphabet Size: {alphabet_size}")

    # Add appropriate plot title, legend, axes labels, etc.
    plt.title("Improvement in Bits vs Multiset Size")
    plt.xlabel("Multiset Size")
    plt.ylabel("Improvement in Bits (%)")
    plt.legend()
    plt.show()
    # for multiset_size in [5, 10, 20, 50, 100, 200, 500, 1000]:
    #     generate_e2e_test(multiset_size)