"""External compressors zstd for testing/benchmarking purposes.

Zstd is a modern LZ77 based compressor, using ANS for encoding match lengths and distances, and
Huffman coding for literals.

Zstd format is described in https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md.

Zstd website: https://github.com/facebook/zstd/.

We use the python zstandard module which internally calls the C library.
"""

import os
import tempfile
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray, bitarray_to_uint, uint_to_bitarray
from scl.utils.test_utils import (
    create_random_binary_file,
    try_file_lossless_compression,
    try_lossless_compression,
)
import zstandard


class ZstdExternalEncoder(DataEncoder):
    def __init__(self, level=6):
        self.level = level
        # state stays alive across blocks so we can benefit
        self.zstd_obj = ZstdExternalEncoder.create_zstd_compressor_object(level=self.level)
        self.block_size_num_bits = 32  # num bits used to encode the block size at start

    def reset(self):
        # start new zstd context
        self.zstd_obj = ZstdExternalEncoder.create_zstd_compressor_object(level=self.level)

    @staticmethod
    def create_zstd_compressor_object(level: int):
        """Create zstd compressor object with similar API as python zlib object.

        Args:
            level (int): Zstd level
        """
        zstd_cctx = zstandard.ZstdCompressor(level=level)
        zstd_obj = zstd_cctx.compressobj()
        return zstd_obj

    def encode_block(self, data_block: DataBlock):
        raw_bytes = bytes(data_block.data_list)

        # flush below with COMPRESSOBJ_FLUSH_BLOCK that ensures decompress is able to decompress the
        # data till now. Note that this still utilizes this block for finding matches when
        # we are compressing the next block (as opposed to COMPRESSOBJ_FLUSH_FINISH that resets the state).
        # See https://python-zstandard.readthedocs.io/en/latest/compressor.html#zstandard.ZstdCompressionObj.flush
        # for more information
        compressed_bytes = self.zstd_obj.compress(raw_bytes) + self.zstd_obj.flush(
            zstandard.COMPRESSOBJ_FLUSH_BLOCK
        )

        # FIXME: might be inefficient to convert to BitArray since it will be later be
        # converted back to bytes when writing to file
        # FIXME: also, should we worry about endianness?

        # at start write the compressed size because zstd decoder expects to get complete blocks
        # and cannot determine when we want to end
        compressed_bitarray = BitArray(
            uint_to_bitarray(len(compressed_bytes) * 8, bit_width=self.block_size_num_bits)
        )
        compressed_bitarray.frombytes(compressed_bytes)
        return compressed_bitarray

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int = 10000):
        """utility wrapper around the encode function using Uint8FileDataStream

        Args:
            input_file_path (str): path of the input file
            encoded_file_path (str): path of the encoded binary file
            block_size (int): choose the block size to be used to call the encode function
        """
        # call the encode function and write to the binary file
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class ZstdExternalDecoder(DataDecoder):
    def __init__(self, level=6):
        self.level = level
        self.zstd_obj = ZstdExternalDecoder.create_zstd_decompressor_object()
        self.block_size_num_bits = 32  # num used to encode the block size at start

    def reset(self):
        self.zstd_obj = ZstdExternalDecoder.create_zstd_decompressor_object()

    @staticmethod
    def create_zstd_decompressor_object():
        """Create zstd compressor object with similar API as python zlib object."""
        zstd_dctx = zstandard.ZstdDecompressor()
        zstd_obj = zstd_dctx.decompressobj()
        return zstd_obj

    def decode_block(self, compressed_bitarray: BitArray):
        # first read the size of the zstd block(s)
        zstd_size = bitarray_to_uint(compressed_bitarray[: self.block_size_num_bits])
        zstd_compressed = compressed_bitarray[
            self.block_size_num_bits : self.block_size_num_bits + zstd_size
        ]

        compressed_bytes = zstd_compressed.tobytes()
        return (
            DataBlock(list(self.zstd_obj.decompress(compressed_bytes))),
            self.block_size_num_bits + zstd_size,
        )

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function using Uint8FileDataStream

        Args:
            encoded_file_path (str): input binary file
            output_file_path (str): output (text) file to which decoded data is written
        """

        # decode data and write output to a text file
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


def test_zstd_encode_decode():
    encoder = ZstdExternalEncoder()
    decoder = ZstdExternalDecoder()

    # create some sample data consisting of bytes
    data_list = [0, 0, 1, 3, 4, 100, 255, 123, 234, 42, 186]
    data_block = DataBlock(data_list)

    is_lossless, _, _ = try_lossless_compression(
        data_block, encoder, decoder, add_extra_bits_to_encoder_output=True
    )
    assert is_lossless


def test_zstd_file_encode_decode():
    """full test for ZstdExternalEncoder and ZstdExternalDecoder

    - create a sample file
    - encode the file using ZstdExternalEncoder
    - perform decoding and check if the compression was lossless

    """
    # define encoder, decoder
    encoder = ZstdExternalEncoder()
    decoder = ZstdExternalDecoder()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a file with some random data
        input_file_path = os.path.join(tmpdirname, "inp_file.txt")
        create_random_binary_file(
            input_file_path,
            file_size=5000,
            prob_dist=ProbabilityDist({44: 0.5, 45: 0.25, 46: 0.2, 255: 0.05}),
        )

        # test lossless compression
        assert try_file_lossless_compression(
            input_file_path, encoder, decoder, encode_block_size=1000
        )
