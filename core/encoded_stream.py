from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint
import tempfile
import os

#################


class Padder:
    NUM_PAD_BITS = 3

    @classmethod
    def add_byte_padding(cls, payload_bitarray: BitArray) -> BitArray:
        assert isinstance(payload_bitarray, BitArray)
        payload_size = len(payload_bitarray)
        num_pad = (8 - (payload_size + cls.NUM_PAD_BITS) % 8) % 8

        padding_bitarray = uint_to_bitarray(num_pad, bit_width=cls.NUM_PAD_BITS) + BitArray(
            "0" * num_pad
        )
        return padding_bitarray + payload_bitarray

    @classmethod
    def remove_byte_padding(cls, payload_pad_bitarray: BitArray) -> BitArray:
        assert isinstance(payload_pad_bitarray, BitArray)
        # get padding
        pad_bitarray = payload_pad_bitarray[: cls.NUM_PAD_BITS]
        num_pad = bitarray_to_uint(pad_bitarray)

        # header
        payload_bitarray = payload_pad_bitarray[cls.NUM_PAD_BITS + num_pad :]
        return payload_bitarray


def test_padder():
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
    NUM_HEADER_BYTES = 4
    NUM_HEADER_BITS = NUM_HEADER_BYTES * 8
    MAX_PAYLOAD_SIZE = 1 << NUM_HEADER_BITS

    @classmethod
    def add_header(cls, payload_bitarray: BitArray):
        # check if bitarray is byte aligned
        assert len(payload_bitarray) % 8 == 0

        # add header
        arr_size = len(payload_bitarray) // 8
        assert arr_size < cls.MAX_PAYLOAD_SIZE  # maximum size of the  block
        header_bitarray = uint_to_bitarray(arr_size, bit_width=cls.NUM_HEADER_BITS)
        return header_bitarray + payload_bitarray

    @classmethod
    def get_payload_size(cls, header_bytes: bytes):
        assert isinstance(header_bytes, bytes)
        header_bitarray = BitArray()
        header_bitarray.frombytes(header_bytes)
        assert len(header_bitarray) == cls.NUM_HEADER_BITS
        return bitarray_to_uint(header_bitarray)  # in bytes


def test_header():
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
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __enter__(self):
        self.file_reader = open(self.file_path, "wb")  # open binary file
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_reader.close()

    def write_block(self, encoded_block: BitArray):
        assert isinstance(encoded_block, BitArray)

        # add padding
        padded_encoded_block = Padder.add_byte_padding(encoded_block)

        # add header
        payload_header_block = HeaderHandler.add_header(padded_encoded_block)

        # get bytes
        payload_bytes = payload_header_block.tobytes()

        # write to file
        self.file_reader.write(payload_bytes)


class EncodedBlockReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def __enter__(self):
        self.file_reader = open(self.file_path, "rb")  # open binary file
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_reader.close()

    def get_block(self):
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

        # remove padding
        payload_bitarray = Padder.remove_byte_padding(payload_pad_bitarray)
        return payload_bitarray


###################################


def test_encoded_block_reader_writer():

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, "tmp_file.txt")

        # create 3 blocks
        payload_bitarray_blocks = [
            BitArray("101000101010111"),
            BitArray("1" * 24),
            BitArray("0" * 20),
        ]
        with EncodedBlockWriter(temp_file_path) as encode_writer:
            for block in payload_bitarray_blocks:
                encode_writer.write_block(block)

        # read in data
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
