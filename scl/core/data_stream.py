import abc
import tempfile
import os
import typing
from scl.core.data_block import DataBlock

Symbol = typing.Any


class DataStream(abc.ABC):
    """abstract class to represent a Data Stream

    The DataStream facilitates the block interface.
    From the interface standpoint, the two functions which are useful are:
    - get_block(block_size) -> returns a DataBlock of the given block_size from the stream
    - write_block(block) -> writes the block of data to the stream

    The DataStream can act as a stream object for both writing and reading blocks
    The two more useful sub-classes of the abstract class are FileDataStream and ListDataStream.
    (see their description for more details)
    """

    @abc.abstractmethod
    def seek(self, pos: int):
        """seek a particular position in the data stream"""
        pass

    @abc.abstractmethod
    def get_symbol(self):
        """returns a symbol from the data stream, returns None if the stream is finished

        This is an abstract method, and hence needs to be implemented by the subclasses
        """
        pass

    def get_block(self, block_size: int) -> DataBlock:
        """returns a block of data (of the given max size) from the stream

        get_block function tries to return a block of size `block_size`.
        In case the remaining stream is shorter, a smaller block will be returned

        Args:
            block_size (int): the (max) size of the block of data to be returned.

        Returns:
            DataBlock:
        """
        # NOTE: we implement get_block as a loop over get_symbol function
        # this is not the most optimal way of imeplemting get_block (as reading a block of data at once might be faster)
        # TODO: investigate faster ways of directly reading a block
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
        """writes the given symbol to the stream

        The symbol can be appropriately converted to a particular format before writing.
        This is an abstract method and so, the subclass will have to implement it

        Args:
            s (Any): symbol to be written to the stream
        """
        pass

    def write_block(self, data_block: DataBlock):
        """write the input block to the stream

        Args:
            data_block (DataBlock): block to be written to the stream
        """
        # NOTE: we implement write_block as a loop over write_symbol function
        # this is not the most optimal way of imeplemting write_block (as writing a block of data at once might be faster)
        # TODO: investigate faster ways of directly writing a block
        for s in data_block.data_list:
            self.write_symbol(s)

    def __enter__(self):
        """function executed while opening the context

        See: https://realpython.com/python-with-statement/. More details in FileDataStream.__enter__ docstring
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Function executed which exiting the context

        Note that the arguments exc_type, exc_value, exc_traceback are as required by python for a context
        """
        pass


class ListDataStream(DataStream):
    """
    ListDataStream is a wrapper around a list of symbols.
    It is useful to:
    - extract data from the list block by block
    - write data to the list block by block

    In practice, this class might be used mainly for testing
    (as usually you would read data from a file.. see FileDataStream for that)
    """

    def __init__(self, input_list: typing.List):
        """initialize with input_list and reset the stream

        Args:
            input_list (List): the list of symbols, around which the class is a wrapper

        Usage:
            with ListDataStream(input_list) as ds:
                block = ds.get_block(block_size=5)
                # do something with the block
        """
        # assert whether the input_list is indeed a list
        assert isinstance(input_list, list)
        self.input_list = input_list

        # set the position counter
        self.current_ind = 0

    def seek(self, pos: int):
        """set the current_ind to a particular pos"""

        assert pos <= len(self.input_list)
        self.current_ind = pos

    def get_symbol(self) -> Symbol:
        """returns the next symbol from the self.input_list"""

        # retrieve the next symbol
        if self.current_ind >= len(self.input_list):
            return None
        s = self.input_list[self.current_ind]

        # increment the current_ind counter
        self.current_ind += 1
        return s

    def write_symbol(self, s: Symbol):
        """write a symbol to the stream"""
        assert self.current_ind <= len(self.input_list)

        # the case where we modify a symbol
        if self.current_ind < len(self.input_list):
            self.input_list[self.current_ind] = s
        else:
            # case where we append a symbol
            self.input_list.append(s)


class FileDataStream(DataStream):
    """Abstract class to create a data stream from a File

    The FileDataStream defines __exit__, __enter__ methods on top of DataStream.
    These methods handle file obj opening/closing

    Subclasses (eg: TextDataStream) need to imeplement methods get_symbol, write_symbol
    to get a functional object.
    """

    def __init__(self, file_path: str, permissions="r"):
        """Initialize the FileDataStream object

        Args:
            file_path (str): path of the file to read from/write to
            permissions (str, optional): Permissions to open the file obj. Use "r" to read, "w" to write to
            (other pyhton file obj permissions also can be used). Defaults to "r".
        """
        self.file_path = file_path
        self.permissions = permissions

    def __enter__(self):
        """open the file object context based on the permissions specified

        NOTE: One way of cleanly managing resources in python is using the with statement
        as shown in the example below. This ensures the resource is released when exiting the context.

        One way to support allow using with statement is defining __enter__ and __exit__ statements,
        which allow for executing functions while entering or exiting the context.
        Reference: https://realpython.com/python-with-statement/

        Example:
        with TextFileDataStream(path, "w") as fds:
            # get a text block
            block = fds.get_block(5)

        """
        self.file_obj = open(self.file_path, self.permissions)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """close the file object at the end of context

        please take a look __enter__ docstring for more info.
        Reference: https://realpython.com/python-with-statement/
        """
        self.file_obj.close()

    def seek(self, pos: int):
        """resets the file object to the beginning"""
        self.file_obj.seek(pos)


class TextFileDataStream(FileDataStream):
    """FileDataStream to read/write text data"""

    def get_symbol(self):
        """get the next character from the text file

        as we read character data from file by default, the get_symbol function does not need to do anything special
        conversions

        Returns:
            (str, None): the next character, None if we reached the end of stream
        """
        s = self.file_obj.read(1)
        if not s:
            return None
        return s

    def write_symbol(self, s):
        """write a character to the text file"""
        self.file_obj.write(s)


class Uint8FileDataStream(FileDataStream):
    """reads Uint8 numbers written to a file"""

    def get_symbol(self):
        """get the next byte from the text file as 8-bit unsigned int

        Returns:
            (int, None): the next byte, None if we reached the end of stream
        """
        s = self.file_obj.read(1)
        if not s:
            return None
        # byteorder doesn't really matter because we just have a single byte
        int_val = int.from_bytes(s, byteorder="big")
        assert 0 <= int_val <= 255
        return int_val

    def write_symbol(self, s):
        """write an 8-bit unsigned int to the text file"""
        assert 0 <= s <= 255
        self.file_obj.write(bytes([s]))


#################################


def test_list_data_stream():
    """simple testing function to check if list data stream is getting generated correctly"""
    input_list = list(range(10))
    with ListDataStream(input_list) as ds:
        for i in range(3):
            block = ds.get_block(block_size=3)
            assert block.size == 3

        block = ds.get_block(block_size=2)
        assert block.size == 1

        block = ds.get_block(block_size=2)
        assert block is None

        # try seeking and reading
        ds.seek(7)
        block = ds.get_block(block_size=5)
        assert block.size == 3
        assert block.data_list[0] == 7

        # try seeking and writing
        ds.seek(5)
        ds.write_symbol(-1)
        block = ds.get_block(block_size=5)
        assert block.size == 5
        assert block.data_list[0] == -1


def test_file_data_stream():
    """function to test file data stream"""

    # create a temporary file
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, "tmp_file.txt")

        # write data to the file
        data_gt = DataBlock(list("This-is_a_test_file"))
        with TextFileDataStream(temp_file_path, "w") as fds:
            fds.write_block(data_gt)

            # try seeking to correct symbol at pos 4
            fds.seek(4)
            fds.write_symbol("_")

        # read data from the file
        with TextFileDataStream(temp_file_path, "r") as fds:
            block = fds.get_block(block_size=4)
            assert block.size == 4

            # try seeking and reading
            fds.seek(4)
            block = fds.get_block(block_size=4)
            assert block.data_list[0] == "_"


def test_uint8_file_data_stream():
    """function to test file data stream"""

    # create a temporary file
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_file_path = os.path.join(tmpdirname, "tmp_file.txt")

        # write data to the file
        data_gt = DataBlock([5, 2, 255, 0, 43, 34])
        with Uint8FileDataStream(temp_file_path, "wb") as fds:
            fds.write_block(data_gt)

            # try seeking to correct symbol at pos 4
            fds.seek(4)
            fds.write_symbol(99)

        # read data from the file
        with Uint8FileDataStream(temp_file_path, "rb") as fds:
            block = fds.get_block(block_size=4)
            assert block.size == 4
            assert block.data_list == [5, 2, 255, 0]

            # try seeking and reading
            fds.seek(4)
            block = fds.get_block(block_size=4)
            assert block.data_list == [99, 34]
