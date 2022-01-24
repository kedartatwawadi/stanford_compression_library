

# wrapper around the data
# This abstraction is useful as we need to know the datatype etc. 
from codecs import EncodedFile
from re import L
from typing import List
import abc
from compressors.huffman_coder import HuffmanTree

from core.data_stream import BitsDataStream

class DataBlock:
    pass

# dataCompressor
class DataCompressor:

    def __init__(self):
        self.encoder_state = {}
        self.decoder_state = {}

    @staticmethod
    @abc.abstractmethod
    def _encode_block(params_dict, data_input, state):
        # This is a static method, so that there is no information "leak"
        # into the class

        pass
        # add code here
        # return data_bits_block

    @final
    def encode_block(self, data_input: List[DataBlock]):
        output_bits_stream = self._encode_block(self.__dict__, data_input, state = None)
        
        # the final output of encoder needs to be a stream of bits
        assert isinstance(output_bits_stream, BitsDataStream)
        return output_bits_stream
    
    @final
    def encode(self, input_stream, output_stream, block_size=None):
        state = None
        while True:
            # create blocks form data_input
            data_block = input_stream.get_block(block_size=block_size)
            if data_block is None:
                break
            output, state = self.encode_block(data_block, state)
            output_stream.write(output)
        
        
    @staticmethod
    @abc.abstractmethod
    def _decode_block(params_dict, data_input, state):
        # This is a static method, so that there is no information "leak"
        # into the class

        pass
        # add code here
        # return the decoded data

    @final
    def decode_block(self, data_input: BitsDataBlock, state = None):
        
        # input stream to the decoder needs to be a stream of bits
        assert isinstance(data_input, BitsDataBlock)
        return self._decode(self.__dict__, data_input, state)
    
    @final
    def encode(self, encoded_stream, decoded_stream):
        state = None
        while True:
            # create blocks form data_input
            encoded_block = encoded_stream.get_block()
            if encoded_block is None:
                break
            decoded_block, state = self.decode_block(encoded_block, state)
            output_stream.write(output)
        



# Huffman encoder with the params given
class HuffmanCoder1(DataCompressor):
    def __init__(self, prob_dict):
        self.prob_dict = prob_dict

    def _encode(params, data_input):
        huff_tree = HuffmanTree(params.prob_dict)
        encoding_table = huff_tree.get_encoding_table()

        # encode 
        encoded_bits = blah blah
        return encoded_bits

    def _decode(params, data_input):
        # add code to decode bits

        huff_tree = HuffmanTree(params.prob_dict)
        decoded_block = blah blah
        return decoded_block

# HuffmanCoder without any params
class HuffmanCoder2(DataCompressor):
    def __init__(self):
        self.prob_dist_compressor = blah blah 

    def _encode(params, data_input):
        # parse the input and infer prob_dict
        prob_dict = data_input.get_empirical_distribution()
        huff_coder = HuffmanCoder1(prob_dict)

        # encode using huff_coder
        encoded_bits = huff_coder.encode(data_input)

        # encode params
        encoded_params = params.prob_dist_compressor.encode(prob_dist)

        # concat and return
        return concat_bit_streams(encoded_params, encoded_bits)

    def _decode(params, data_input):
        # split bit stream
        encoded_params, encoded_bits = split_bit_streams(data_input)

        # decode prob_dict
        prob_dict = params.prob_dist_compressor.decode(encoded_params)

        # decode data
        huff_coder = HuffmanCoder1(prob_dict)
        data_decoded = huff_coder.decode(encoded_bits)

        return data_decoded


# Usage: 
# r = RunLengthCoder()
# bits = r.encode(data_block)
# decoded_block = r.decode(bits)
class RunLengthCoder(DataCompressor):
    """
    Simple run length encoder which splits input into symbols and run lengths and encodes them separately

    """
    def __init__(self):
        self.symbols_compressor = HuffmanCoder2()
        self.run_length_compressor = GolombCoder()

    def _encode(params, data_input):
        symbol_block, run_length_block =  run_length_parser.forward(data_input)

        # encode symbols
        encoded_symbols = params.symbols_compressor.encode(symbols_block)

        # encode run_length
        encoded_run_lengths = params.run_length_compressor.encode(run_length_block) 

        # concat
        return concat_bit_streams(encoded_symbols, encoded_run_lengths)


    def _decode(params, data_input):
        # split
        encoded_symbols, encoded_run_lengths = split_bit_streams(encoded_symbols, encoded_run_lengths)

        # decode symbols
        decoded_symbols = params.symbols_compressor.decode(encoded_symbols)

        # decode run_lengths
        decoded_run_lengths = params.run_length_compressor.decode(encoded_run_lengths)

        # combine symbols and run_lengths
        data_decoded = run_length_parser.inverse(decoded_symbols, decoded_run_lengths)






class DataCompressor(abc.ABC):

    class Encoder:
        def encode_block(self, data_block):
            # update state, return bits
            # self.state = ...
            raise NotImplementedError

        @final
        def encode(self, data_stream, output_stream, block_size):
            # combine bits together from block
            while True:
                # create blocks form data_input
                data_block = data_stream.get_block(block_size)

                # if data_block is None, we done, so break
                if data_block is None:
                    break

                # encode and return state
                output = self.encode_block(data_block)
                assert isinstance(output, BitsDataBlock)
                output_stream.write(output)
        
    class Decoder:
        def decode_block(self, bits_block):
            # update state, return decoded_data
            # self.state = ...
            raise NotImplementedError

        @final
        def decode(self, encoded_stream, output_stream):
            # combine bits together from block
            while True:
                # create blocks form data_input
                encoded_block = encoded_stream.get_block()

                # if data_block is None, we done, so break
                if encoded_block is None:
                    break

                # encode and return state
                output = self.decode_block(encoded_block)
                output_stream.write(output)
    
    @final
    def __init__(self, *args, **kwargs):
        # define encoder
        # same inputs are passed to encoder decoder for initialization
        self._encoder = DataCompressor.Encoder(*args, **kwargs)
        self._decoder = DataCompressor.Decoder(*args, **kwargs)

        # map encode, encode_block ... functions to nested classes
        self.encode = self._encoder.encode
        self.decode = self._decoder.decode
        self.encode_block = self._encoder.encode_block
        self.decode_block = self._decoder.decode_block


class HuffmanCoder1(DataCompressor):

    class Encoder(DataCompressor.Encoder):
        def __init__(self, prob_dict=prob_dict):
            huffman_tree = HuffmanTree(prob_dict)
            self.encoding_table = huffman_tree.get_encoding_table()
            # do something with the params

        def encode_block(self, data_block):
            # encode
            bits_block = lookup_table_func(data_block)
            return bits_block
    
    
    class Decoder(DataCompressor.Decoder):
        def __init__(self, prob_dict=prob_dict):
            self.huffman_tree = HuffmanTree(prob_dict)

        def decode_block(self, encoded_bits_block):
            # decode
            return decoded_block


class HuffmanCoder2(DataCompressor):

    class Encoder(DataCompressor.Encoder):
        def __init__(self):

            # this is the "state" we save
            self.empirical_counts = {}

            self.prob_dist_compressor = ...

        def encode_block(self, data_block):
            counts = data_block.get_counts()
            #update counts to get a prob_dist
            self.empirical_counts = ...

            prob_dist = ...

            huff_coder = HuffmanCoder1(prob_dist)
            
            # encode using huff_coder
            encoded_bits = huff_coder.encode(data_block)

            # encode params
            encoded_params = self.prob_dist_compressor.encode(prob_dist)

            # concat and return
            return concat_bit_streams(encoded_params, encoded_bits)



    



