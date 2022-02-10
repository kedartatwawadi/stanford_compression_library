import abc
import tempfile
import os
from typing import List
from core.data_block import DataBlock


class DataStream(abc.ABC):
    """
    Class to represent the input stream
    """

    @abc.abstractmethod
    def reset(self):
        # resets the data stream
        pass

    @abc.abstractmethod
    def get_symbol(self):
        pass  # returns None if the stream is finished

    def get_data_block(self, block_size: int):
        # returns the next data block
        data_list = []
        for _ in range(block_size):
            # get next symbol
            s = self.get_symbol()
            if s is None:
                break
            data_list.append(s)

        # if data_list is empty, return None to signal the stream is over
        if not data_list:
            return None

        return DataBlock(data_list)

    @abc.abstractmethod
    def write_symbol(self, s):
        pass

    def write_data_block(self, data_block: DataBlock):
        for s in data_block.data_list:
            self.write_symbol(s)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


class ListDataStream(DataStream):
    """
    create a data stream object from a list
    """

    def __init__(self, input_list: List):
        # assert whether the input_list is indeed a list
        assert isinstance(input_list, list)
        self.input_list = input_list

        # reset counter
        self.reset()

    def reset(self):
        self.current_ind = 0

    def get_symbol(self):
        if self.current_ind >= len(self.input_list):
            return None
        s = self.input_list[self.current_ind]
        self.current_ind += 1
        return s

    def write_symbol(self, s):
        self.input_list.append(s)


class FileDataStream(DataStream):
    """
    create a data stream object from a file
    """

    def __init__(self, file_path: str, permissions="r"):
        """
        block class -> specifies what type of block to return
        Also, every DataBlock has a char_to_symbol function which is used to convert data appropriately before passing
        """
        self.file_path = file_path
        self.permissions = permissions

    def __enter__(self):
        self.file_obj = open(self.file_path, self.permissions)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file_obj.close()

    def reset(self):
        self.file_obj.seek(0)


class TextFileDataStream(FileDataStream):
    """
    reads one symbol at a time
    """

    def get_symbol(self):
        s = self.file_obj.read(1)
        if not s:
            return None
        return s

    def write_symbol(self, s):
        self.file_obj.write(s)


class Uint8FileDataStream(FileDataStream):
    """
    reads Uint8 numbers written to a file
    """

    pass


def test_list_data_stream():
    """
    simple testing function to check if list data stream is getting generated correctly
    """
    input_list = list(range(10))
    with ListDataStream(input_list) as ds:
        for i in range(3):
            block = ds.get_data_block(block_size=3)
            assert block.size == 3

        block = ds.get_data_block(block_size=2)
        assert block.size == 1

        block = ds.get_data_block(block_size=2)
        assert block is None


def test_file_data_stream():
    """
    function to test file data stream
    """

    # create a temporary file
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, "tmp_file.txt")

        # write data to the file
        data_gt = DataBlock(list("This_is_a_test_file"))
        with TextFileDataStream(temp_file_path, "w") as fds:
            fds.write_data_block(data_gt)

        # read data from the file
        with TextFileDataStream(temp_file_path, "r") as fds:
            block = fds.get_data_block(block_size=4)
            assert block.size == 4
