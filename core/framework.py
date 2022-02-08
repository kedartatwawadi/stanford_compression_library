import abc

from core.data_block import BitsDataBlock, DataBlock


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
    def encode(self, data_stream: DataStream, output_stream, block_size: int):
        # combine bits together from block

        self.reset()

        while True:
            # create blocks form data_input
            data_block = data_stream.get_block(block_size, pad=False)

            # if data_block is None, we done, so break
            if data_block is None:
                break

            # encode and return state
            output = self.encode_block(data_block)
            assert isinstance(output, BitsDataBlock)
            output_stream.write(output)


class DataDecoder(abc.ABC):
    def __init__(self):
        self.state = {}

    def reset(self):
        self.state = {}

    def decode_block(self, bits_block):
        # update state, return decoded_data
        # self.state = ...
        raise NotImplementedError

    @final
    def decode(self, encoded_stream, output_stream):
        # combine bits together from block

        # reset the state
        self.reset()

        while True:
            # create blocks form data_input
            encoded_block = encoded_stream.get_block()

            # if data_block is None, we done, so break
            if encoded_block is None:
                break

            # encode and return state
            output = self.decode_block(encoded_block)
            output_stream.write(output)
