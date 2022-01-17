import unittest
from core.data_block import DataBlock, StringDataBlock
from core.data_transformer import IdentityTransformer, SplitStringTransformer


class IdentityTransformerTest(unittest.TestCase):
    def test_identity(self):
        data_list = [0, "A", "B", 1]
        data_block = DataBlock(data_list)
        output_block = IdentityTransformer().transform(data_block)

        assert output_block.size == data_block.size
        for id in range(output_block.size):
            assert data_block.data_list[id] == output_block.data_list[id]


class SplitStringTransformerTest(unittest.TestCase):
    def test_validate_func_on_invalid_input(self):
        """
        checks the validation func of the StringDataBlock class
        """
        data_list = ["0", "A1", "BA1", "B1CD"]
        data_block = StringDataBlock(data_list)
        output_block = SplitStringTransformer().transform(data_block)
        assert output_block.size == 10
        assert output_block.data_list[-1] == "D"
