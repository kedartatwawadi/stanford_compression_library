from scl.compressors.probability_models import (
    AdaptiveIIDFreqModel,
    AdaptiveOrderKFreqModel,
    FreqModelBase,
)
from scl.core.data_block import DataBlock
from scl.core.prob_dist import Frequencies
from scl.utils.test_utils import try_lossless_compression
import copy
from scl.compressors.arithmetic_coding import AECParams, ArithmeticDecoder, ArithmeticEncoder
import numpy as np


def pseudo_random_LFSR_generator(data_size, tap, noise_prob=0, seed=0):
    """
    Generate a pseudo-random sequence using a LFSR
    :param data_size: size of the output sequence
    :param tap: tap of the LFSR, e.g. 4 for x[n] = x[n-1] xor x[n-4]
    :param noise_prob: probability of Bernoulli noise
    :return: output_sequence of length data_size
    """
    np.random.seed(seed)
    # initial sequence = [1,0,0,0,...]
    initial_sequence = [0] * tap
    initial_sequence[0] = 1

    # output sequence
    output_sequence = initial_sequence
    for _ in range(data_size - tap):
        s = output_sequence[-1] ^ output_sequence[-tap]  # xor
        if noise_prob > 0:
            s = s ^ np.random.binomial(1, noise_prob)
        output_sequence.append(s)
    return output_sequence


def convert_float_prob_to_int(p, M=1000):
    """
    Convert a float probability to an integer probability
    :param p: float probability
    :param M: multiplier
    :return: integer probability
    """
    assert 0 <= p <= 1, "p must be between 0 and 1"
    return int(p * M)


class NoisyLFSRFreqModel:
    """
    A frequency model for a noisy LFSR
    """

    def __init__(self, tap: int, noise_prob: float):
        self.tap = tap
        self.noise_prob = noise_prob
        self.freqs_current = Frequencies({0: 1, 1: 1})  # initializes the freqs with uniform probability

        # maintain last few symbols
        self.past_few_symbols = []

    def update_model(self, s):
        """
        Updates the freq model (`self.freqs_current`) based on the next symbol s
        :param s: symbol to be encoded next
        """
        # ###############################
        # ADD CODE HERE FOR: updating self.freqs_current, this is the probability
        # distribution to be used for next symbol.
        # Check the implementation in `AdaptiveIIDFreqModel` and
        # `AdaptiveOrderKFreqModel` in scl.compressors.probability_models for inspiration
        # HINTS:
        # (1) you need to keep track of the context, in this case past self.tap number of symbols.
        # we are saving the past TAP symbols in a buffer: self.past_few_symbols.
        # You may use these to update the model and to set `self.freqs_current`
        # (2) you should use the `Frequencies` class to create the probability distribution;
        # frequencies are stored as integers for float probability values as we have seen in class
        # (e.g., 1000 for 0.001) and probabilities are invariant to the total frequency, so you can set the
        # total frequency to any value you want. For the autograder purposes, you can assume noise_prob to be in
        # multiples of 0.001 (e.g., 0.001, 0.002, 0.003, etc.), i.e. noise_prob = 0.001 * noise_prob_int.
        # You can also use the helper function `convert_float_prob_to_int` to convert a float probability to a valid int
        raise NotImplementedError
        ###############################

        aec_params = AECParams() # params used for arithmetic coding in SCL
        assert self.freqs_current.total_freq <= aec_params.MAX_ALLOWED_TOTAL_FREQ, (
            f"Total freq {self.freqs_current.total_freq} is greater than "
            f"max allowed total freq {aec_params.MAX_ALLOWED_TOTAL_FREQ} for arithmetic coding in SCL. This leads to"
            f"precision and speed issues. Try reducing the total freq by a factor of 2 or more."
        )
        self.freqs_current._validate_freq_dist(self.freqs_current.freq_dict) # check if freqs are valid datatype


def test_adaptive_order_k_arithmetic_coding():
    """
    Test the adaptive order k arithmetic coding with various contexts
    """
    # for the source as defined in _generate_2nd_order_markov, the
    # 0th and 1st order entropy is log_2(3) [uniform] and beyond that it is 1.
    # data_block = DataBlock(pseudo_random_LFSR_generator(10000, initial_sequence=[1,0,0,0], taps=[-1,-4]))
    DATA_SIZE = 10000
    noise_prob = 0.01
    # set to True to run the LFSR model after implementation of the `update_model` function,
    # to see reported results in the questions, set to False
    run_LFSR_model = True
    seed = 0

    for TAP in [3, 4, 7, 15, 22]:
        data_block = DataBlock(
            pseudo_random_LFSR_generator(DATA_SIZE, tap=TAP, noise_prob=noise_prob, seed=seed)
        )

        # print newline so it shows up nicely on testing
        print()
        print("-" * 10)
        if noise_prob > 0:
            print(
                f"Data generated as: X[n] = X[n-1] \u2295 X[n-{TAP}] \u2295 Bern_noise({noise_prob})"
            )
        else:
            print(f"Data generated as: X[n] = X[n-1] \u2295 X[n-{TAP}]")
        
        print(f"DATA_SIZE={DATA_SIZE}")

        for k in [0, 1, 2, 3, 4, 7, 15, 22, 24]:
            # define AEC params
            aec_params = AECParams()

            # define encoder/decoder models
            # NOTE: important to make a copy, as the encoder updates the model, and we don't want to pass
            # the update model around
            freq_model_enc = AdaptiveOrderKFreqModel(
                alphabet=list(range(2)),
                k=k,
                max_allowed_total_freq=aec_params.MAX_ALLOWED_TOTAL_FREQ,
            )
            freq_model_dec = copy.deepcopy(freq_model_enc)

            # create encoder/decoder
            encoder = ArithmeticEncoder(aec_params, freq_model_enc)
            decoder = ArithmeticDecoder(aec_params, freq_model_dec)

            # check if encoding/decoding is lossless
            is_lossless, encode_len, _ = try_lossless_compression(
                data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
            )

            assert is_lossless

            # avg codelen ignoring the bits used to signal num data elements
            avg_codelen = encode_len / data_block.size
            print(f"AdaptiveArithmeticCoding, k={k}, avg_codelen: {avg_codelen:.3f}")

        if run_LFSR_model:
            # define AEC params
            aec_params = AECParams()
            # define encoder/decoder models
            # NOTE: important to make a copy, as the encoder updates the model, and we don't want to pass
            # the update model around
            freq_model_enc = NoisyLFSRFreqModel(tap=TAP, noise_prob=noise_prob)
            freq_model_dec = copy.deepcopy(freq_model_enc)

            # create encoder/decoder
            encoder = ArithmeticEncoder(aec_params, freq_model_enc)
            decoder = ArithmeticDecoder(aec_params, freq_model_dec)

            # check if encoding/decoding is lossless
            is_lossless, encode_len, _ = try_lossless_compression(
                data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
            )

            assert is_lossless

            # avg codelen ignoring the bits used to signal num data elements
            avg_codelen = encode_len / data_block.size
            print(f"LFSR Model, k={k}, avg_codelen: {avg_codelen:.3f}")
