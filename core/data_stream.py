import abc
import tempfile
import os
from core.data_block import DataBlock
from typing import List


class DataStream(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        # resets the data stream
        pass

    @abc.abstractmethod
    def get_next_data_block(self, block_size):
        # returns the next data block
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


class ListDataStream(DataStream):
    """
    create a data stream object from a list
    """

    def __init__(self, input_list: List, block_cls=DataBlock):
        # assert whether the input_list is indeed a list
        assert isinstance(input_list, list)
        self.input_list = input_list

        # store the block_cls to initialize the output with
        self.block_cls = block_cls

        # reset the stream
        self.reset()

    def reset(self):
        self.start_ind = 0

    def get_next_data_block(self, block_size):
        if block_size is None:
            return self.input_list

        # return None if you have reached the end of list
        if self.start_ind >= (len(self.input_list)):
            return None

        end = min(self.start_ind + block_size, len(self.input_list))
        data = self.input_list[self.start_ind : end]
        self.start_ind += block_size

        # We assume data is already formatted correctly in the input list
        return self.block_cls(data)


class FileDataStream(DataStream):
    """
    create a data stream object from a file
    """

    def __init__(self, file_path: str, block_cls=DataBlock):
        """
        block class -> specifies what type of block to return
        Also, every DataBlock has a char_to_symbol function which is used to convert data appropriately before passing
        """
        self.file_path = file_path

        # store the block_cls to initialize the output with
        self.block_cls = block_cls

    def __enter__(self):
        self.file_reader = open(self.file_path, "r")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_reader.close()

    def reset(self):
        self.file_reader.seek(0)

    def get_next_data_block(self, block_size):
        assert block_size is not None

        # get raw data
        data_raw = self.file_reader.read(block_size)
        if data_raw == "":
            return None

        # format data appropriately based on what type of datablock we want
        data = [self.block_cls.char_to_symbol(c) for c in data_raw]
        return self.block_cls(data)


def test_list_data_stream():
    """
    simple testing function to check if list data stream is getting generated correctly
    """
    input_list = list(range(10))
    with ListDataStream(input_list) as ds:
        for i in range(3):
            block = ds.get_next_data_block(block_size=3)
            assert block.size == 3

        block = ds.get_next_data_block(block_size=2)
        assert block.size == 1

        block = ds.get_next_data_block(block_size=2)
        assert block is None


def test_file_data_stream():
    """
    function to test file data stream
    """

    # create a temporary file
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, "tmp_file.txt")

        # write data to the file
        data_gt = "This_is_a_test_file"
        with open(temp_file_path, "w") as fp:
            fp.write(data_gt)

        # read data from the file
        with FileDataStream(temp_file_path) as fds:
            block = fds.get_next_data_block(block_size=4)
            assert block.size == 4
