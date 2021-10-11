import unittest
from core.data_stream import DataStream, StringDataStream
from core.data_transformer import IdentityTransformer, SplitStringTransformer


class IdentityTransformerTest(unittest.TestCase):
    """
    checks basic operations for a DataStream
    FIXME: improve these tests
    """

    def test_identity(self):
        data_list = [0, "A", "B", 1]
        data_stream = DataStream(data_list)
        output_stream = IdentityTransformer().transform(data_stream)

        assert output_stream.size == data_stream.size
        for id in range(output_stream.size):
            assert data_stream.data_list[id] == output_stream.data_list[id]


class SplitStringTransformerTest(unittest.TestCase):
    """
    checks the validation func of the StringDataStream class
    """

    def test_validate_func_on_invalid_input(self):
        data_list = ["0", "A1", "BA1", "B1CD"]
        data_stream = StringDataStream(data_list)
        output_stream = SplitStringTransformer().transform(data_stream)
        assert output_stream.size == 10
        assert output_stream.data_list[-1] == "D"
