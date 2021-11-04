import numpy as np
from core.data_stream import BitsDataStream, DataStream
from core.data_transformer import DataTransformer


class DataCompressor:
    """
    Base Data Compressor
    """

    def __init__(
        self, encoder_transform: DataTransformer = None, decoder_transform: DataTransformer = None
    ):
        self.encoder_transform = encoder_transform
        self.decoder_transform = decoder_transform

    def set_encoder_decoder_params(self, data_stream: DataStream):
        """
        Usually we will set the self.encoder_transform and self.decoder_transform in this function
        as most of time the parameters of the data_stream are necessary to create the transformers
        FIXME: This is a bit ugly
        """
        pass

    def encode(self, data_stream: DataStream):
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

    def decode(self, data_stream: BitsDataStream):

        # input stream to the decoder needs to be a stream of bits
        assert isinstance(data_stream, BitsDataStream)
        return self.decoder_transform.transform(data_stream)
