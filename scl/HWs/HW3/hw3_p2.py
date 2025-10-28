from scl.core.data_block import DataBlock
from scl.core.data_stream import TextFileDataStream

class BurrowsWheelerTransform:
    def __init__(self, delimiter="~"):
        # NOTE: we assume delimiter not present in the input
        self.delimiter = delimiter

    def forward(self, data_block: DataBlock):
        """
        Generates the forward transform of BWT
        NOTE: for consistency all forward and inverse functions take in as input
        a DataBlock
        """

        # create a string using data_block
        input_block_str = "".join(data_block.data_list)

        ###############################################
        # ADD CODE HERE
        # to generate bwt_str (BWT transformed string)
        # Note: remember to add the delimiter to the string!
        raise NotImplementedError("Implement this function")
        ###############################################

        data_bwt_block = DataBlock(list(bwt_str))
        return data_bwt_block

    def inverse(self, bwt_block: DataBlock):
        """
        Generates the inverse of the BWT.
        NOTE: for consistency all forward and inverse functions take in as input
        a DataBlock
        """
        bwt_block_str = "".join(bwt_block.data_list)
        N = len(bwt_block_str)

        ###############################################
        # ADD CODE HERE
        # to generate output_str
        # Note: remember to remove the delimiter from the string!
        raise NotImplementedError("Implement this function")
        ###############################################

        return DataBlock(list(output_str))


class MoveToFrontTransform:
    def __init__(self):
        # NOTE: this table should work for this HW
        self.table = [chr(c) for c in range(128)]

    def forward(self, data_block):
        table = self.table.copy()
        output_data_list = []
        for c in data_block.data_list:
            rank = table.index(c)  # Find the rank of the character in the dictionary [O(k)]
            output_data_list.append(rank)  # Update the encoded text

            # Update the table
            table.pop(rank)
            table.insert(0, c)
        return DataBlock(output_data_list)

    def inverse(self, data_block_mtf):
        decoded_data_list = []
        table = self.table.copy()

        for rank in data_block_mtf.data_list:
            c = table[rank]
            decoded_data_list.append(c)

            # Update the dictionary
            table.pop(rank)
            table.insert(0, c)

        return DataBlock(decoded_data_list)


def test_bwt_transform():
    bwt_transform = BurrowsWheelerTransform()

    sample_inputs = ["BANANA", "abracadabraabracadabraabracadabra", "hakunamatata"]
    expected_bwt_outputs = ["BNN~AAA", "rrdd~aadrrrcccraaaaaaaaaaaabbbbbba", "hnmtt~aauaaka"]
    for sample_input, expected_bwt_str in zip(sample_inputs, expected_bwt_outputs):
        print("\n" + "-" * 20)
        print(f"Input string: {sample_input}")

        # Get the BWT transformed string
        block = DataBlock(list(sample_input))
        bwt_block = bwt_transform.forward(block)
        bwt_str = "".join(bwt_block.data_list)
        print(f"BWT transfomed string: {bwt_str}")
        assert bwt_str == expected_bwt_str

        # get the inverse BWT
        inv_bwt = bwt_transform.inverse(bwt_block)
        inv_bwt_str = "".join(inv_bwt.data_list)
        print(f"I-BWT: {inv_bwt_str}")
        assert sample_input == inv_bwt_str


def test_mtf_transform():
    mtf = MoveToFrontTransform()

    sample_inputs = [
        "BANANA",
        "BNN~AAA",
        "abracadabraabracadabraabracadabra",
        "rrdd~aadrrrcccraaaaaaaaaaaabbbbbba"
    ]
    for sample_input in sample_inputs:

        # create MTF forward transforms for the given strings
        block = DataBlock(list(sample_input))
        mtf_block = mtf.forward(block)
        print("\n" + "-" * 20)
        print(f"Input str: {sample_input}")
        print(f"MTF: {mtf_block.data_list}")

        inv_mtf = mtf.inverse(mtf_block)
        inv_mtf_str = "".join(inv_mtf.data_list)
        print(f"MTF inverse: {inv_mtf_str}")
        assert inv_mtf_str == sample_input


def test_bwt_mtf_entropy():
    DATA_BLOCK_SIZE = 50000
    FILE_PATH = "sherlock_ascii.txt"

    # read in DATA_BLOCK_SIZE bytes
    with TextFileDataStream(FILE_PATH, "r") as fds:
        data_block = fds.get_block(block_size=DATA_BLOCK_SIZE)
    print()
    print(f"Input data: 0-order Empirical Entropy: {data_block.get_entropy():.4f}")

    bwt = BurrowsWheelerTransform()
    data_bwt_transformed = bwt.forward(data_block)
    # print(''.join(data_block.data_list))
    # print(''.join(data_bwt_transformed.data_list))
    print(f"Input data + BWT: 0-order Empirical Entropy: {data_bwt_transformed.get_entropy():.4f}")

    mtf = MoveToFrontTransform()
    data_mtf_transformed = mtf.forward(data_block)
    print(f"Input data + MTF: 0-order Empirical Entropy: {data_mtf_transformed.get_entropy():.4f}")

    bwt = BurrowsWheelerTransform()
    mtf = MoveToFrontTransform()
    data_bwt_transformed = bwt.forward(data_block)
    data_bwt_mtf_transformed = mtf.forward(data_bwt_transformed)
    # print(data_bwt_mtf_transformed.data_list)
    print(f"Input data + BWT + MTF: 0-order Empirical Entropy: {data_bwt_mtf_transformed.get_entropy():.4f}")

