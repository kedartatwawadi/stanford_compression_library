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
    First example of a simple compressor.
    Encodes each symbol of the data to a fixed bitwidth
    """

    def set_encoder_decoder_params(self, data_stream):
        """
        TODO: add stuff here
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
        self.encoder_transform = CascadeTransformer(
            [
                LookupTableTransformer(self.encoder_lookup_table),
                UintToBitstringTransformer(bit_width=self.bit_width),
                BitstringToBitsTransformer(),
            ]
        )

        # create decoder transform
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
