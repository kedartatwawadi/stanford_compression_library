import abc
from typing import final
from core.data_block import DataBlock
from core.data_stream import DataStream
from core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
from core.util import BitArray


class DataEncoder(abc.ABC):
    def __init__(self):
        self.state = {}

    def reset(self):
        self.state = {}

    def encode_block(self, data_block: DataBlock):
        # update state, return bits
        # self.state = ...
        raise NotImplementedError

    @final
    def encode(self, data_stream: DataStream, block_size: int, encode_writer: EncodedBlockWriter):
        # combine bits together from block

        self.reset()

        while True:
            # create blocks form data_input
            data_block = data_stream.get_block(block_size)

            # if data_block is None, we are done, so break
            if data_block is None:
                break

            # encode the block
            output = self.encode_block(data_block)
            assert isinstance(output, BitArray)
            encode_writer.write_block(output)


class DataDecoder(abc.ABC):
    def __init__(self):
        self.state = {}

    def reset(self):
        self.state = {}

    def decode_block(self, bitarray: BitArray):
        # update state, return decoded_data
        # self.state = ...
        raise NotImplementedError

    @final
    def decode(self, encode_reader: EncodedBlockReader, output_stream: DataStream):
        # combine bits together from block

        # reset the state
        self.reset()

        while True:
            encoded_block = encode_reader.get_block()

            # if encode_block is None, we done, so break
            if encoded_block is None:
                break

            # encode and return state
            output = self.decode_block(encoded_block)
            output_stream.write_block(output)
