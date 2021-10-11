import numpy as np
from core.data_stream import BitsDataStream
from core.data_transformer import (
    BitsToBitstringTransformer,
    BitstringToBitsTransformer,
    BitstringToUintTransformer,
    CascadeTransformer,
    DataTransformer,
    LookupTableTransformer,
    UintToBitstringTransformer,
)


class DataCompressor:
    """
    Base Data Compressor
    """

    def __init__(
        self, encoder_transform: DataTransformer = None, decoder_transform: DataTransformer = None
    ):
        self.encoder_transform = encoder_transform
        self.decoder_transform = decoder_transform

    def set_encoder_decoder_params(self, data_stream):
        """
        Usually we will set the self.encoder_transform and self.decoder_transform in this function
        as most of time the parameters of the data_stream are necessary to create the transformers
        FIXME: This is a bit ugly
        """
        pass

    def encode(self, data_stream):
        """
        The core encode function of the compressor
        """

        # set the parameters of the encoder/decoder using the data_stream
        self.set_encoder_decoder_params(data_stream)

        # perform the encoding
        output_bits_stream = self.encoder_transform.transform(data_stream)

        # the final output of encoder needs to be a stream of bits
        assert isinstance(output_bits_stream, BitsDataStream)
        return output_bits_stream

    def decode(self, data_stream):

        # input stream to the decoder needs to be a stream of bits
        assert isinstance(data_stream, BitsDataStream)
        return self.decoder_transform.transform(data_stream)


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
