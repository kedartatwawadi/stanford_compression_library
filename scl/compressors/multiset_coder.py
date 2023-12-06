import numpy as np
import random
from scl.utils.multiset_utils import MultiSetNode
from scl.utils.gen_json_maps import gen_dog_info
from scl.utils.gen_floats import gen_floats
from math import log2, factorial, ceil
import json
from tqdm import tqdm

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


def run_json_map_e2e_test(num_keys, num_json_entries):
    data = gen_dog_info(num_keys=num_keys, num_json_entries=num_json_entries)
    big_multiset = MultiSetNode.from_iterable(MultiSetNode.from_iterable(item.items()) for item in data)
    multiset = MultiSetNode.from_iterable(MultiSetNode.from_iterable(item.items()) for item in data)
    multiset_size = len(big_multiset)

    print("Generated initial multiset and backup")

    null_terminator = "\0"

    upper_ascii_bound = 128

    def ascii_encode(state, symbol):
        assert ord(symbol) < upper_ascii_bound
        return rans_encode(state, ord(symbol), 1, upper_ascii_bound)

    def ascii_decode(state):
        symbol = rans_decode_slot(state, upper_ascii_bound)
        state = rans_decode(state, symbol, 1, upper_ascii_bound)
        return state, symbol

    def symbol_encode(state, symbol):
        key, value = symbol
        state = ascii_encode(state, null_terminator)
        for char in key:
            state = ascii_encode(state, char)
        state = ascii_encode(state, null_terminator)
        for char in value:
            state = ascii_encode(state, char)
        # print("encoded:", key, value)
        return state

    def symbol_decode(state):
        value = []
        while True:
            state, symbol = ascii_decode(state)
            if symbol == ord(null_terminator):
                break
            value.insert(0, chr(symbol))

        key = []
        while True:
            state, symbol = ascii_decode(state)
            if symbol == ord(null_terminator):
                break
            key.insert(0, chr(symbol))
        # print("decoded:", key, value)
        return state, ("".join(key), "".join(value))

    def nested_multiset_encode(state: int, multiset: MultiSetNode):
        multiset.verify()
        for _ in tqdm(range(multiset_size)):
            assert state >= initial_state
            # Sample symbol
            # print("Current multiset state:", multiset)
            state, inner_multiset = swor_decode(state, multiset)
            # print("Removed inner multiset:", inner_multiset)
            # print("Post removal:", multiset)
            multiset.verify()

            # Encode symbol
            state = multiset_encode(state, inner_multiset.clone(), symbol_encode)
        assert multiset.empty
        return state

    def nested_multiset_decode(state: int):
        multiset = MultiSetNode()
        for _ in tqdm(range(multiset_size)):
            # Decode symbol
            state, inner_multiset = multiset_decode(state, num_keys, symbol_decode)
            # print("Decoded multiset:", inner_multiset)

            # Encode back bits for sampling
            state = swor_encode(state, inner_multiset, multiset)
        return state, multiset


    print("Multiset size:", multiset_size)

    initial_state = 202347390
    encoded_state = nested_multiset_encode(initial_state, multiset)
    print("Encoded state size:", ceil(log2(encoded_state)))

    output_state, decoded_multiset = nested_multiset_decode(encoded_state)
    assert initial_state == output_state

    def serialize_multiset(multiset: MultiSetNode):
        outputs = []
        for item in multiset:
            cur_json = {}
            for (key, value) in item:
                cur_json[key] = value
            outputs.append(json.dumps(cur_json, sort_keys=True))

        return sorted(outputs)

    # print("Decoded multiset:", decoded_multiset)
    assert big_multiset == decoded_multiset
    # ser1 = serialize_multiset(big_multiset)
    # ser2 = serialize_multiset(decoded_multiset)
    # with open("ser1.txt", "w") as f:
    #     f.write("\n".join(ser1))
    # with open("ser2.txt", "w") as f:
    #     f.write("\n".join(ser2))
    # assert ser1 == ser2

    state = initial_state
    for item in tqdm(data):
        for kv in item.items():
            state = symbol_encode(state, kv)

    baseline_size = ceil(log2(state))
    print("Baseline size:", baseline_size)

    outputs = []
    for _ in tqdm(range(multiset_size)):
        items = []
        for _ in range(num_keys):
            state, kv = symbol_decode(state)
            items.append(kv)

        multiset = MultiSetNode.from_iterable(items)
        outputs.append(multiset)

    assert initial_state == state
    assert MultiSetNode.from_iterable(outputs) ==  big_multiset

    actual_bits_used = ceil(log2(encoded_state))
    baseline_bits_used = baseline_size
    theoretical_bits_saved = log2(factorial(num_keys)) * multiset_size + log2(factorial(multiset_size))
    theoretical_bits_used = baseline_bits_used - theoretical_bits_saved

    return {
        "percentage_savings": 100 * ((baseline_bits_used - actual_bits_used) / baseline_bits_used),
        "theoretical_bits_used": theoretical_bits_used,
        "actual_bits_used": actual_bits_used,
        "bits_saved": baseline_bits_used - actual_bits_used,
        "theoretical_bits_saved": baseline_bits_used - theoretical_bits_used,
        "percentage_of_theoretical_saved": 100 * ((baseline_bits_used - actual_bits_used) / (baseline_bits_used - theoretical_bits_used))
    }


def test_json_map():
    num_keys = 3

    # Plot percentage savings for different sizes
    import matplotlib.pyplot as plt
    outputs = []
    sizes = [50, 250, 500, 1000]
    for num_entries in sizes:
        outputs.append(run_json_map_e2e_test(num_keys, num_json_entries=num_entries))

    # Plot percentage savings for different sizes
    x = sizes
    y = [output["percentage_savings"] for output in outputs]

    plt.plot(x, y)
    plt.xlabel("Number of Entries")
    plt.ylabel("Percentage Savings")
    plt.title("Percentage Savings vs Number of Entries")
    plt.show()





def run_float_e2e_test(float_type, negative_proportion, num_floats):
    float_len = len(float_type().tobytes())
    data = gen_floats(float_type, negative_proportion=negative_proportion, num_floats=num_floats)
    # print(data)
    print("Length of each float:", float_len)
    num_floats = len(data)

    multiset = MultiSetNode.from_iterable(int(item >= 0) * 2 - 1 for item in data)
    orig_multiset = multiset.clone()

    def byte_encode(state, symbol):
        return rans_encode(state, symbol, 1, 256)

    def byte_decode(state):
        symbol = rans_decode_slot(state, 256)
        state = rans_decode(state, symbol, 1, 256)
        return state, symbol

    def symbol_encode(state, symbol):
        b = symbol.tobytes()
        # print("Encoding:", symbol, list(b))
        for i in range(float_len - 1, -1, -1):
            state = byte_encode(state, b[i])
        return state

    def symbol_decode(state):
        buffer = []
        for _ in range(float_len):
            state, b = byte_decode(state)
            buffer.append(b)
        # print("Decoding:", buffer)
        return state, np.frombuffer(bytes(buffer), dtype=float_type)

    def floats_encode(state: int, floats):
        for value in tqdm(floats):
            # Sample sign
            state, sign = swor_decode(state, multiset)

            # Encode symbol
            abs_value = np.abs(value)
            # assert len(value.tobytes()) == len(abs_value.tobytes()) == float_len
            # print(len(np.float32(sign).tobytes()))
            # assert float_len == len((np.multiply(abs_value, sign)).tobytes())
            state = symbol_encode(state, abs_value * float_type(sign))
        return state

    def floats_decode(state: int):
        multiset = MultiSetNode()
        outputs = []
        for _ in tqdm(range(num_floats)):
            # Decode symbol
            state, value = symbol_decode(state)
            value = value[0]
            outputs.append(value)
            # print("Decoded value:", value)
            sign = int(value >= 0) * 2 - 1

            # Encode back bits for sampling
            state = swor_encode(state, sign, multiset)
        return state, multiset, outputs

    initial_state = 202347390
    encoded_state = floats_encode(initial_state, data)
    print("Encoded state size:", ceil(log2(encoded_state)))

    output_state, decoded_multiset, decoded_floats = floats_decode(encoded_state)
    assert initial_state == output_state

    assert orig_multiset == decoded_multiset

    assert sorted(abs(x) for x in data) == sorted(abs(x) for x in decoded_floats)

    state = initial_state
    for value in data:
        state = symbol_encode(state, value)

    baseline_size = ceil(log2(state))
    print("Baseline size:", baseline_size)

    outputs = []
    for _ in tqdm(range(num_floats)):
        state, symbol = symbol_decode(state)
        outputs.append(symbol)

    assert state == initial_state

    theoretical_bits_used = (float_len * 8 - 1) * num_floats
    actual_bits_used = ceil(log2(encoded_state))
    baseline_bits_used = baseline_size

    # Return percentage savings over theoretical, theoretical bits saved, and actual bits used in a dictionary
    return {
        "percentage_savings": 100 * ((baseline_bits_used - actual_bits_used) / baseline_bits_used),
        "theoretical_bits_used": theoretical_bits_used,
        "actual_bits_used": actual_bits_used,
        "bits_saved": baseline_bits_used - actual_bits_used,
        "theoretical_bits_saved": baseline_bits_used - theoretical_bits_used,
        "percentage_of_theoretical_saved": 100 * ((baseline_bits_used - actual_bits_used) / (baseline_bits_used - theoretical_bits_used))
    }


def test_float_sign():
    float_type = np.float64
    num_floats = 1000
    negative_proportion = 0.25

    result = run_float_e2e_test(float_type, negative_proportion, num_floats)
    print(result)

    # Using matplotlib, plot the % savings in bits over numpy 16, 32, 64 bit floats nicely with different lines, colors, and a plot title, legend, and labeled axes
    import matplotlib.pyplot as plt
    for float_type in [np.float16, np.float32, np.float64]:
        result = run_float_e2e_test(float_type, negative_proportion, num_floats)
        print(result)

        # Plot in matplotlib with a scatter plot and x-axis as float size and y-axis as % savings
        plt.scatter(len(float_type().tobytes()) * 8, result["percentage_savings"], label=f"np.{float_type.__name__}")

    # Add appropriate plot title, legend, axes labels, etc.
    plt.title("Percentage Savings vs Float Size")
    plt.xlabel("Theoretical Bits Used")
    plt.ylabel("Percentage Savings")
    plt.legend()
    plt.show()




