import unittest
from core.data_stream import DataStream, StringDataStream
from core.data_transformer import IdentityTransformer, SplitStringTransformer


class IdentityTransformerTest(unittest.TestCase):
    def test_identity(self):
        data_list = [0, "A", "B", 1]
        data_stream = DataStream(data_list)
        output_stream = IdentityTransformer().transform(data_stream)

        assert output_stream.size == data_stream.size
        for id in range(output_stream.size):
            assert data_stream.data_list[id] == output_stream.data_list[id]


class SplitStringTransformerTest(unittest.TestCase):
    def test_validate_func_on_invalid_input(self):
        """
        checks the validation func of the StringDataStream class
        """
        data_list = ["0", "A1", "BA1", "B1CD"]
        data_stream = StringDataStream(data_list)
        output_stream = SplitStringTransformer().transform(data_stream)
        assert output_stream.size == 10
        assert output_stream.data_list[-1] == "D"
