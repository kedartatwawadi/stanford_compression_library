"""encoded stream writers and readers

All the encoders and decoders in the SCL process data in blocks.
The encoder can encode data_block to a differently sized encoded_bitarray. To allow the decoder to retrieve the encoded_bitarray
correcponding to a data_block, we have to include some block header before each encoded_bitarray
The EncodedBlockWriter and EncodedBlockReader handle adding/removing these block headers and providing an interface to just
care about the encoded_bitarray. More information in the respective docstrings.
"""

from scl.utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
import tempfile
import os
import pytest

#################


class Padder:
    """Class to byte pad a bitarray"""

    NUM_PAD_BITS = 3

    @classmethod
    def add_byte_padding(cls, payload_bitarray: BitArray) -> BitArray:
        """add byte padding to payload_bitarray

        - the padding size is represented using the first 3 bits (0-7)
        - the structure of the returned bitarray is:
          (num_pad_bits + padding + payload)
        - as we are essentially adding 3 bits to the payload, the padding computation needs to take that into account
            Args:
                payload_bitarray (BitArray): input bitarray

        Returns:
            BitArray: padded bitarray
        """
        if not isinstance(payload_bitarray, BitArray):
            raise TypeError("payload must be a BitArray")

        # compute how much padding to add
        payload_size = len(payload_bitarray)
        num_pad = (8 - (payload_size + cls.NUM_PAD_BITS) % 8) % 8

        # add padding
        padding_bitarray = uint_to_bitarray(num_pad, bit_width=cls.NUM_PAD_BITS) + BitArray(
            "0" * num_pad
        )
        return padding_bitarray + payload_bitarray

    @classmethod
    def remove_byte_padding(cls, payload_pad_bitarray: BitArray) -> BitArray:
        """Remove byte padding, rejecting missing headers or nonzero padding bits."""
        if not isinstance(payload_pad_bitarray, BitArray):
            raise TypeError("padded payload must be a BitArray")
        if len(payload_pad_bitarray) < 8 or len(payload_pad_bitarray) % 8:
            raise ValueError("padded payload must contain at least one complete byte")
        # get padding
        pad_bitarray = payload_pad_bitarray[: cls.NUM_PAD_BITS]
        num_pad = bitarray_to_uint(pad_bitarray)
        payload_start = cls.NUM_PAD_BITS + num_pad
        if payload_start > len(payload_pad_bitarray):
            raise ValueError("padding length exceeds the padded payload")
        if any(payload_pad_bitarray[cls.NUM_PAD_BITS : payload_start]):
            raise ValueError("padding bits must be zero")

        # header
        payload_bitarray = payload_pad_bitarray[payload_start:]
        return payload_bitarray


def test_padder():
    """test adding/removing byte padding to payload"""

    def _test(bits_gt):
        # add padding
        padded_bits_gt = Padder.add_byte_padding(bits_gt)
        assert len(padded_bits_gt) % 8 == 0

        # remove padding
        padding_removed_bits = Padder.remove_byte_padding(padded_bits_gt)
        assert bits_gt == padding_removed_bits

    payloads = [BitArray("1" * size) for size in range(32)]
    for payload in payloads:
        _test(payload)


#################


class HeaderHandler:
    """Add block header to an encoded bitarray

    The header communicates the size of the encoded bitarray
    - the header itself is a fixed 4 bytes in size, and communicates the length in bytes of the payload
    - returned bitarray -> header (32 bits) + payload
    """

    NUM_HEADER_BYTES = 4
    NUM_HEADER_BITS = NUM_HEADER_BYTES * 8
    MAX_PAYLOAD_SIZE = 1 << NUM_HEADER_BITS

    @classmethod
    def add_header(cls, payload_bitarray: BitArray):
        """add header to the byte aligned payload bitarray"""
        # check if bitarray is byte aligned
        if not isinstance(payload_bitarray, BitArray):
            raise TypeError("payload must be a BitArray")
        if len(payload_bitarray) % 8:
            raise ValueError("payload must be byte aligned")

        # add header
        arr_size = len(payload_bitarray) // 8
        if arr_size >= cls.MAX_PAYLOAD_SIZE:
            raise ValueError("payload is too large for the block header")
        header_bitarray = uint_to_bitarray(arr_size, bit_width=cls.NUM_HEADER_BITS)
        return header_bitarray + payload_bitarray

    @classmethod
    def get_payload_size(cls, header_bytes: bytes):
        """returns the size of the payload by reading in the header"""
        if not isinstance(header_bytes, bytes):
            raise TypeError("header must be bytes")
        if len(header_bytes) != cls.NUM_HEADER_BYTES:
            raise ValueError("block header must contain exactly four bytes")
        header_bitarray = BitArray()
        header_bitarray.frombytes(header_bytes)
        return bitarray_to_uint(header_bitarray)  # in bytes


def test_header():
    """tests the header"""
    payload_bitarray = BitArray("1" * 23)
    padded_payload_bitarray = Padder.add_byte_padding(payload_bitarray)

    # get true size of the padded payload
    size_payload = len(padded_payload_bitarray) // 8

    # add header
    payload_header_bitarray = HeaderHandler.add_header(padded_payload_bitarray)
    data_bytes = payload_header_bitarray.tobytes()

    # decode size
    size_payload_decoded = HeaderHandler.get_payload_size(
        data_bytes[: HeaderHandler.NUM_HEADER_BYTES]
    )
    assert size_payload == size_payload_decoded


######################################


class EncodedBlockWriter:
    """writer to write bitarrays to the encoded file"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __enter__(self):
        self.file_writer = open(self.file_path, "wb")  # open binary file
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_writer.close()

    def write_block(self, encoded_block: BitArray):
        """writes the encoded_block to the file

        Lets say we have two encoded bitarrays of variable length: enc_block1, enc_block2
        the EncodedBlockWriter does the following things to each enc_block
        - input -> (enc_block)
        - Padding -> byte align enc_block (padding + enc_block)
        - Header -> add a block header. The block header communicates the size of the enc_block
        (header + padding + enc_block)

        Args:
            encoded_block (BitArray): encoded bitarray to be written to the file
        """

        # check if the encoded_block is indeed a BitArray
        if not isinstance(encoded_block, BitArray):
            raise TypeError("encoded block must be a BitArray")

        # add byte padding
        padded_encoded_block = Padder.add_byte_padding(encoded_block)

        # add block header
        payload_header_block = HeaderHandler.add_header(padded_encoded_block)

        # write to file
        payload_bytes = payload_header_block.tobytes()
        self.file_writer.write(payload_bytes)


class EncodedBlockReader:
    """Reader to read encoded_blocks from a compressed binary file"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def __enter__(self):
        self.file_reader = open(self.file_path, "rb")  # open binary file
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_reader.close()

    def get_block(self):
        """read the next encoded_block from the encoded file

        The encoded file contains bits written in blocks such as:
        (header + padding + enc_block1), (header + padding + enc_block2)
        The reader needs to infer where the blocks end:
        - header -> header size is fixed, so read the header and infer the (padding + enc_block) size
        - padding -> remove byte padding from the payload (enc_block) and return the payload

        Returns:
            payload_bitarray (BitArray): the encoded bitarray block
        """
        # read header
        header_bytes = self.file_reader.read(HeaderHandler.NUM_HEADER_BYTES)
        if len(header_bytes) == 0:
            # reached end of file
            return None

        # header size should be correct
        if len(header_bytes) != HeaderHandler.NUM_HEADER_BYTES:
            raise ValueError("truncated encoded block header")

        # get block size (in bytes)
        payload_size = HeaderHandler.get_payload_size(header_bytes)

        # read the payload
        payload_bytes = self.file_reader.read(payload_size)
        if len(payload_bytes) != payload_size:
            raise ValueError("truncated encoded block payload")

        # construct payload bitarray and remove padding
        payload_pad_bitarray = BitArray()
        payload_pad_bitarray.frombytes(payload_bytes)

        # remove byte padding
        payload_bitarray = Padder.remove_byte_padding(payload_pad_bitarray)
        return payload_bitarray


###################################


def test_encoded_block_reader_writer():
    """tests EncodedBlockReader and EncodedBlockWriter

    Testing procedure:
    - create some dummy encoded_blocks
    - write the encoded blocks to a binary file using EncodedBlockWriter
    - read the binary file back using EncodedBlockReader and assert if the data is the same
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create 3 dummy encoded blocks
        payload_bitarray_blocks = [
            BitArray(),
            BitArray("101000101010111"),
            BitArray("1" * 24),
            BitArray("0" * 20),
            BitArray(),
        ]

        # write the encoded blocks to a binary file
        temp_file_path = os.path.join(tmpdirname, "encoded.bin")
        with EncodedBlockWriter(temp_file_path) as encode_writer:
            for block in payload_bitarray_blocks:
                encode_writer.write_block(block)

        # read back the encoded blocks
        encoded_blocks = []
        with EncodedBlockReader(temp_file_path) as encode_reader:
            while True:
                block = encode_reader.get_block()
                if block is None:
                    break
                encoded_blocks.append(block)

        # check if the read data is equal to the data written
        assert len(payload_bitarray_blocks) == len(encoded_blocks)
        for written_block, read_block in zip(payload_bitarray_blocks, encoded_blocks):
            assert written_block == read_block


@pytest.mark.parametrize(
    "encoded_hex",
    [
        "00",  # partial size header
        "000000",  # partial size header
        "00000000",  # no padding header
        "00000002a0",  # declared payload is longer than the file
        "00000001e0",  # seven pad bits cannot fit after a three-bit header
        "00000001a1",  # padding bits must be zero
    ],
)
def test_encoded_block_reader_rejects_invalid_framing(tmp_path, encoded_hex):
    encoded_path = tmp_path / "invalid.bin"
    encoded_path.write_bytes(bytes.fromhex(encoded_hex))
    with EncodedBlockReader(str(encoded_path)) as reader:
        with pytest.raises(ValueError):
            reader.get_block()


def test_encoded_block_reader_distinguishes_empty_payload_from_eof(tmp_path):
    encoded_path = tmp_path / "empty.bin"
    encoded_path.write_bytes(bytes.fromhex("00000001a0"))
    with EncodedBlockReader(str(encoded_path)) as reader:
        assert reader.get_block() == BitArray()
        assert reader.get_block() is None


@pytest.mark.parametrize("bits", ["", "101", "1010000", "101000000"])
def test_padder_rejects_incomplete_bytes(bits):
    with pytest.raises(ValueError):
        Padder.remove_byte_padding(BitArray(bits))


def test_header_rejects_invalid_input():
    with pytest.raises(ValueError):
        HeaderHandler.add_header(BitArray("1"))
    with pytest.raises(TypeError):
        HeaderHandler.add_header(b"\x00")
    with pytest.raises(TypeError):
        HeaderHandler.get_payload_size(BitArray("0" * 32))
    for header_size in [0, 1, 3, 5]:
        with pytest.raises(ValueError):
            HeaderHandler.get_payload_size(b"\x00" * header_size)
