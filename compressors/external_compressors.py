"""Standard compressors like gzip, etc. for testing/benchmarking purposes
"""

from core.data_block import DataBlock
from utils.bitarray_utils import BitArray
from utils.test_utils import try_lossless_compression
import zlib

# TODO - add a flag to enable/disable adding size header to front

class ZlibExternalEncoder:
    def __init__(self, level=6):
        self.level = level
        # state stays alive across blocks so we can benefit 
        self.state = {"zlib_context": zlib.compressobj(level=self.level)}

    def encode_block(self, data_block: DataBlock):
        if "zlib_context" not in self.state:
            # e.g., if we reset at some point
            self.state["zlib_context"] = zlib.compressobj(level=self.level)
        zlib_context = self.state["zlib_context"]
        raw_bytes = bytes(data_block.data_list)
        # flush below with Z_SYNC_FLUSH that ensures decompress is able to decompress the 
        # data till now. Note that this still utilizes this block for finding matches when
        # we are compressing the next block (as opposed to Z_FULL_FLUSH that resets the state).
        # See https://www.zlib.net/manual.html for more information
        compressed_bytes = zlib_context.compress(raw_bytes) + zlib_context.flush(zlib.Z_SYNC_FLUSH)
        # inefficient to convert to BitArray since it will be later be
        # converted back to bytes when writing to file
        compressed_bitarray = BitArray()  # worry about endianness??
        compressed_bitarray.frombytes(compressed_bytes)
        return compressed_bitarray


class ZlibExternalDecoder:
    def __init__(self, level=6):
        self.level = level
        self.state = {"zlib_context": zlib.decompressobj()}

    def decode_block(self, compressed_bitarray: BitArray):
        if "zlib_context" not in self.state:
            # e.g., if we reset at some point
            self.state["zlib_context"] = zlib.decompressobj()
        zlib_context = self.state["zlib_context"]
        compressed_bytes = compressed_bitarray.tobytes()
        return DataBlock(list(zlib_context.decompress(compressed_bytes))), len(compressed_bitarray)


def test_zlib_encode_decode():
    encoder = ZlibExternalEncoder()
    decoder = ZlibExternalDecoder()

    # create some sample data consisting of bytes
    data_list = [0, 0, 1, 3, 4, 100, 255, 123, 234, 42, 186]
    data_block = DataBlock(data_list)

    is_lossless, _, _ = try_lossless_compression(data_block, encoder, decoder)
    assert is_lossless