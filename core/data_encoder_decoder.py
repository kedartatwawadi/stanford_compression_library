"""defines DataEncoder and DataDecoder classes

DataEncoder and DataDecoder are the base classes for all encoders, decoders
They impelement a lot of utility functions which make it easier to implement differene encoders/decoders.
More info in respective docstrings
"""

import abc
from core.data_block import DataBlock
from core.data_stream import DataStream, TextFileDataStream
from core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
from utils.bitarray_utils import BitArray


class DataEncoder(abc.ABC):
    """base abstract class for imeplementing any data encoder

    - any subclassing encoder needs to only implement encode_block function, which just operates on a give block of data
    - the appropriate concatenation of encoded blocks is handled by encode function, and need not be re-imeplemented by
    subclasses
    """

    def reset(self):
        """reset the state if any"""
        # NOTE: the user can call this to clear any state the encoder might 
        # have persisted across encode_block calls
        pass

    def encode_block(self, data_block: DataBlock):
        """Abstract class to encode a given block of data

        Subclassing encoders need to mainly implement this method

        Args:
            data_block (DataBlock): input data_block

        Returns:
            encoded_bitarray (BitArray): the encoded bitarray
        """
        # return encoded_bitarray
        raise NotImplementedError

    
    def encode(self, data_stream: DataStream, block_size: int, encode_writer: EncodedBlockWriter):
        """function to encode a given data_stream

        - chops the data_stream into blocks (specified by block_size)
        - each data_block is encoded to a encoded_bitarray by the self.encoded_block function.
        - the encoded_bitarray is then written to a output binary file using the encode_writer

        Args:
            data_stream (DataStream): input data stream
            block_size (int): the block size used to chop the input data stream
            encode_writer (EncodedBlockWriter): the writer used to write encoded bitarrays
        """

        while True:
            # create blocks form data_input
            data_block = data_stream.get_block(block_size)

            # if data_block is None, we are done, so break
            if data_block is None:
                break

            # encode the block
            output = self.encode_block(data_block)
            assert isinstance(output, BitArray)

            # write encoded bitarrays
            encode_writer.write_block(output)

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int = 10000):
        """utility wrapper around the encode function

        NOTE: This function assumes the input_file_path is a text file, as is to be aprsed using the
        TextFileDataStream. This is the most common behavior; but certain compressors can overwrite this method to
        support other files as input (for example files containing numerical data)

        Args:
            input_file_path (str): path of the input file
            encoded_file_path (str): path of the encoded binary file
            block_size (int): choose the block size to be used to call the encode function
        """
        # call the encode function and write to the binary file
        with TextFileDataStream(input_file_path, "r") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class DataDecoder(abc.ABC):
    """abstract class used to define a decoder

    - any subclassing decoder needs to mainly implement the decode_block method
    - accessing and decoding one encoded block at a time is handled by the decode function, which need not be re-imeplemnted
    """

    def reset(self):
        """reset the state, if any"""
        # NOTE: the user can call this to clear any state the decoder might 
        # have persisted across decode_block calls
        pass

    def decode_block(self, bitarray: BitArray):
        """abstract function to decode one encoded_bitarray

        subclassing decoders mainly need to only implement this function.

        Args:
            bitarray (BitArray): input encoded bitarray

        Returns:
            decoded_block (DataBlock), num_bits_consumed (int): returns the decoded data and how many bits were used
            the num_bits_consumed can be used appropriately if the encoded bitarray contains encoded bits from more than one encoders
        """

        # return decoded_block, num_bits_consumed
        raise NotImplementedError

    
    def decode(self, encode_reader: EncodedBlockReader, output_stream: DataStream):
        """function to decode a binary encoded stream

        - The binary encoded stream consists of multiple encoded blocks of data.
          `EncodedBlockReader` retrieves data one block at a time
        - the retrieved encoded_block is decoded using the `decode_block` method
        - the decoded data_blocks are written out using the `output_stream`

        Args:
            encode_reader (EncodedBlockReader): EncodedBlockReader object to read blocks of encoded bitarrays
            output_stream (DataStream): DataStream object to write decoded blocks of data
        """

        while True:
            # read the next encoded block
            encoded_block = encode_reader.get_block()

            # if encode_block is None, we reached end of stream, so break
            if encoded_block is None:
                break

            # encode and return state
            output_block, num_bits_consumed = self.decode_block(encoded_block)
            assert num_bits_consumed == len(encoded_block)

            # write decoded blocks to DataStream
            output_stream.write_block(output_block)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function

        NOTE: this utility function assumes that the output DataStream is of type TextFileDataStream.
        This is the most common behavior; but certain compressors can overwrite this method to
        support other files as input (for example files containing numerical data)
        Args:
            encoded_file_path (str): input binary file
            output_file_path (str): output (text) file to which decoded data is written
        """

        # decode data and write output to a text file
        with EncodedBlockReader(encoded_file_path) as reader:
            with TextFileDataStream(output_file_path, "w") as fds:
                self.decode(reader, fds)
