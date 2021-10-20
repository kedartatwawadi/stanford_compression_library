"""
Contains some elementary baseline compressors
1. Fixed bit width compressor 
"""

import numpy as np
from core.data_transformer import (
    BitsToBitstringTransformer,
    BitstringToBitsTransformer,
    BitstringToUintTransformer,
    CascadeTransformer,
    LookupTableTransformer,
    UintToBitstringTransformer,
)
from core.data_compressor import DataCompressor


class FixedBitwidthCompressor(DataCompressor):
    """
    Encodes each symbol of the data to a fixed bitwidth
    Example: data = [A,B,B,A,C]
    Encoding ~ ['00',`01`, `01`, `00`, `10`] -> converted to bits
    See baseline_compressors_tests.py for usage
    """

    def set_encoder_decoder_params(self, data_stream):
        """
        Each compressor consists of encoder_transform + decoder_transform
        In this case we need to know the input alphabet size to define the transforms

        We first get the alphabet size and then set the encoder and decoder transforms
        in this function
        """
        # set bit width
        alphabet = data_stream.get_alphabet()
        self.bit_width = self._get_bit_width(len(alphabet))

        # set encoder/decoder lookup tables
        self.encoder_lookup_table = {}
        self.decoder_lookup_table = {}

        for id, a in enumerate(alphabet):
            self.encoder_lookup_table[a] = id
            self.decoder_lookup_table[id] = a

        # create encoder and decoder transforms
        # The encoder transformer is represented as a cascade:
        # Example:
        # data = [A,B,B,A,C]
        # Transform-1: Lookuptable to get he alphabet index {A: 0, B: 1, C: 2}
        # Transform-2: Uint to bitstring with a fixed width (2 for example) [0 -> 00, 1 -> 01, 2 -> 10]
        # Transform-3: Bitstring to bits: [000 -> 0,0,0]
        self.encoder_transform = CascadeTransformer(
            [
                LookupTableTransformer(self.encoder_lookup_table),
                UintToBitstringTransformer(bit_width=self.bit_width),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
        # The decoder transform is also a cascade, but in the reverse order
        self.decoder_transform = CascadeTransformer(
            [
                BitsToBitstringTransformer(bit_width=self.bit_width),
                BitstringToUintTransformer(),
                LookupTableTransformer(self.decoder_lookup_table),
            ]
        )

    @staticmethod
    def _get_bit_width(alphabet_size) -> int:
        return int(np.ceil(np.log2(alphabet_size)))
