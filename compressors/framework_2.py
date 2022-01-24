

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
        self._encoder = self.Encoder(*args, **kwargs)
        self._decoder = self.Decoder(*args, **kwargs)

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
            encoded_bits = huff_coder.encode_block(data_block)

            # encode params
            encoded_params = self.prob_dist_compressor.encode_block(prob_dist)

            # concat and return
            return concat_bit_streams(encoded_params, encoded_bits)


        def decode_block(...)
            ...
    
