from scl.utils.bitarray_utils import uint_to_bitarray, bitarray_to_uint
from scl.core.data_encoder_decoder import DataBlock, DataEncoder, DataDecoder
from scl.core.prob_dist import Frequencies
from typing import List
import numpy as np


class StatefulEncoder(DataEncoder):
    """A stateful encoder is an encoder which maintains a state while encoding.
    The state is initialized to a particular value and is updated after encoding
    each symbol. The state is flushed after encoding the entire block."""
    def __init__(self, initial_state) -> None:
        super().__init__()
        self.initial_state = initial_state

    def encode_symbol(self, s, state):
        # returns state_new
        raise NotImplementedError

    def encode_block(self, data_block):
        # encode the data_block one symbol at a time
        state = self.initial_state
        for s in data_block.data_list:
            state = self.encode_symbol(s, state)

        # flush the state
        encoded_bitarray = uint_to_bitarray(state)
        return encoded_bitarray


class StatefulDecoder(DataDecoder):
    """A stateful decoder is a decoder which maintains a state while decoding. The
    state is initialized to a particular value and is updated after decoding each
    symbol. The state is flushed after decoding the entire block."""
    def __init__(self, initial_state) -> None:
        self.initial_state = initial_state

    def decode_symbol(self, state):
        # return s (symbol), updated self.state
        raise NotImplementedError

    def decode_block(self, encoded_bitarray):
        state = bitarray_to_uint(encoded_bitarray)

        decoded_list = []
        while state > self.initial_state:
            s, state = self.decode_symbol(state)
            decoded_list.append(s)
        assert state == self.initial_state
        decoded_list.reverse()

        return DataBlock(decoded_list)


#################################################################################

def encode_op(state, s, num_symbols):
    '''
    :param state: state
    :param s: symbol
    :param num_symbols: parameter M in HW write-up
    :return: state_next: updated next state
    '''
    state_next = state * num_symbols + s
    return state_next


def decode_op(state, num_symbols):
    '''
    :param state: state
    :param num_symbols: parameter M in HW write-up
    :return: s, state_prev: symbol and previous state
    '''
    s = state_prev = None
    
    ####################################################
    # ADD CODE HERE
    raise NotImplementedError
    ####################################################

    return s, state_prev


class UniformDistEncoder(StatefulEncoder):
    """Uniform distribution encoder. See question for details."""
    def __init__(self, num_symbols) -> None:
        '''
        :param num_symbols: parameter M in HW write-up
        '''
        super().__init__(initial_state=num_symbols)
        self.num_symbols = num_symbols

    def encode_symbol(self, s, state):
        '''
        :param s: symbol
        :param state: current state
        :return: state: updated state
        '''
        state = encode_op(state, s, self.num_symbols)
        return state


class UniformDistDecoder(StatefulDecoder):
    """Uniform distribution decoder. See question for details."""
    def __init__(self, num_symbols) -> None:
        '''
        :param num_symbols: parameter M in HW write-up
        '''
        super().__init__(initial_state=num_symbols)
        self.num_symbols = num_symbols

    def decode_symbol(self, state):
        '''
        :param state: state to be decoded
        :return: (s, state): symbol and updated state
        '''
        s, state = decode_op(state, self.num_symbols)
        return s, state


#################################################################################

class NonUniformDistEncoder(StatefulEncoder):
    """
    Non-uniform distribution encoder. See question for details.
    """
    def __init__(self, freq: Frequencies) -> None:
        '''
        :param freq: frequency list of the symbols
        '''
        self.freq = freq
        super().__init__(initial_state=self.freq.total_freq)

    def encode_symbol(self, s, state):
        '''
        :param s: symbol
        :param state: current state
        :return: state: updated state
        '''
        # decode a "fake" uniform sample between [0, freq[s]]
        fake_locator_symbol, state = decode_op(state, self.freq.freq_dict[s])
        # print(s, fake_locator_symbol, state)
        # create a new symbol
        combined_symbol = self.freq.cumulative_freq_dict[s] + fake_locator_symbol

        # encode the new symbol
        state = encode_op(state, combined_symbol, self.freq.total_freq)
        # print(state)
        # print("*" * 5)
        return state


class NonUniformDistDecoder(StatefulDecoder):
    """
    Non-uniform distribution decoder. See question for details.
    """
    def __init__(self, freq: Frequencies) -> None:
        '''
        :param freq: frequency list of the symbols
        '''
        self.freq = freq
        super().__init__(initial_state=self.freq.total_freq)

    @staticmethod
    def find_bin(cumulative_freqs_list: List, slot: int) -> int:
        """Performs binary search over cumulative_freqs_list to locate which bin
        the slot lies.

        Args:
            cumulative_freqs_list (List): the sorted list of cumulative frequencies
                For example: freqs_list = [2,7,3], cumulative_freqs_list [0,2,9]
            slot (int): the value to search in the sorted list

        Returns:
            bin: the bin in which the slot lies
        """
        # NOTE: side="right" corresponds to search of type a[i-1] <= t < a[i]
        bin = np.searchsorted(cumulative_freqs_list, slot, side="right") - 1
        return int(bin)

    def decode_symbol(self, state):
        '''
        :param state: current state
        :return: (s, state): symbol and updated state
        '''
        #################################################
        # ADD CODE HERE
        # a few relevant helper functions are implemented:
        # self.find_bin, self.freq.total_freq, self.freq.cumulative_freq_dict, self.freq.freq_dict, self.freq.alphabet

        # Step 1: decode (s, z) using joint distribution; (i) decode combined symbol, (ii) find s
        # Step 2: encode z given s; (i) find fake locator symbol, (ii) encode back the fake locator symbol

        # You should be able to use encode_op, decode_op to encode/decode the uniformly distributed symbols

        raise NotImplementedError
        #################################################

        return s, state


######################################################################################

def test_uniform_coder():
    # tests if the uniform coding is working as expected
    # tests if the expected codelength is indeed ~ log2(num_symbols)
    enc = UniformDistEncoder(10)
    dec = UniformDistDecoder(10)

    N = 10
    X = list(np.random.randint(0, 10, N))

    bits = enc.encode_block(DataBlock(X))
    decoded_data_block = dec.decode_block(bits)

    assert decoded_data_block.data_list == X, "Uniform coder is not working as expected"
    # test almost equal
    assert np.abs(len(bits) - N * np.log2(10)) / N < 1, f"codelegth is not as expected:" \
                                                         f" len(bits) = {len(bits)}, expected = {N * np.log2(10)}"

def test_non_uniform_coder():
    # tests if the decoding is lossless
    freq = Frequencies({"A": 1, "B": 1, "C": 2})
    enc = NonUniformDistEncoder(freq)
    dec = NonUniformDistDecoder(freq)

    bits = enc.encode_block(DataBlock(["A", "A", "B", "B", "A", "C"]))
    decoded_data_block = dec.decode_block(bits)
    # print(decoded_data_block.data_list)

    assert decoded_data_block.data_list == ["A", "A", "B", "B", "A", "C"], \
        "Non-uniform coder is not working as expected"
