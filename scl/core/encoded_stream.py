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
        assert isinstance(payload_bitarray, BitArray)

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
        """remove added byte padding"""
        assert isinstance(payload_pad_bitarray, BitArray)
        # get padding
        pad_bitarray = payload_pad_bitarray[: cls.NUM_PAD_BITS]
        num_pad = bitarray_to_uint(pad_bitarray)

        # header
        payload_bitarray = payload_pad_bitarray[cls.NUM_PAD_BITS + num_pad :]
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

    payloads = [BitArray("10110"), BitArray("1" * 23)]
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
        assert len(payload_bitarray) % 8 == 0

        # add header
        arr_size = len(payload_bitarray) // 8
        assert arr_size < cls.MAX_PAYLOAD_SIZE  # maximum size of the  block
        header_bitarray = uint_to_bitarray(arr_size, bit_width=cls.NUM_HEADER_BITS)
        return header_bitarray + payload_bitarray

    @classmethod
    def get_payload_size(cls, header_bytes: bytes):
        """returns the size of the payload by reading in the header"""
        assert isinstance(header_bytes, bytes)
        header_bitarray = BitArray()
        header_bitarray.frombytes(header_bytes)
        assert len(header_bitarray) == cls.NUM_HEADER_BITS
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
        assert isinstance(encoded_block, BitArray)

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
        assert len(header_bytes) == HeaderHandler.NUM_HEADER_BYTES

        # get block size (in bytes)
        payload_size = HeaderHandler.get_payload_size(header_bytes)

        # read the payload
        payload_bytes = self.file_reader.read(payload_size)
        assert len(payload_bytes) == payload_size

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
            BitArray("101000101010111"),
            BitArray("1" * 24),
            BitArray("0" * 20),
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
