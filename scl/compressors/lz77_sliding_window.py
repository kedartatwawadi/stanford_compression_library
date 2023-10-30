"""
See the doc comments in lz77.py for context and references.

This is an implementation of sliding window LZ77 (unlike lz77.py which works with an
ever-growing window). The encoder and decoder are implemented in LZ77SlidingWindowEncoder
and LZ77SlidingWindowDecoder classes, respectively.

The LZ77Window implements the sliding window using a circular buffer
based on a bytearray. The encoder and decoder receive the window size as a parameter (if the
decoder is initialized with a smaller window size than that used for encoding, that might lead
to failure). The window is responsible for handling all the complexity around the position in window
vs. position in data, and hence the rest of the codebase can simply work with absolute positions in
the file.

Another feature of this implementation compared to LZ77Encoder from lz77.py is that the match finder
is provided as a parameter to the encoder, enabling use to implement and try various strategies
in an elegant way. The MatchFinderBase class has the interface for this, allowing to set the window,
find the best match (this is implemented by specific implementations), and common functionality like
extending the match and generating the match length etc. Currently we include a particular match finder
implementation: HashBasedMatchFinder (see documentation in the class). This improves upon the match finder
in LZ77Encoder from lz77.py in a few ways: (i) fixing a maximum hash table size and chain length, 
(ii) left extension of matches in case of hash misses (part of base class), (iii) Lazy match finding. These
techniques allow the HashBasedMatchFinder to obtain better results by using smaller hash lengths.

Note that LZ77SlidingWindowEncoder is compatible with LZ77Decoder (from lz77.py) (see tests).
But LZ77Decoder will not be able to use the limited window and hence will use more resources than
strictly required.

The LZ77Sequence and LZ77StreamsEncoder/LZ77StreamsDecoder classes are reused as is from lz77.py. In future we 
can consider having the entropy coder also be a free parameter.

Future optimizations (some of these will break compatibility with LZ77Decoder unless we update that also):
- have window size be encoded as part of the format.
- add support for repcodes (used in zstd and LZMA, the idea being to reserve some offsets to denote
  recently seen offsets - this improves compression ratio and speed for structured data where multiple
  matches can be found at the same offset, interspered by non-matching segments).
- add more advanced match-finding strategies.

Benchmarks on a few files from https://github.com/nemequ/squash-corpus and
https://corpus.canterbury.ac.nz/descriptions/#cantrbry (plus a few handmade).
Comparing to gzip, zstd and LZ77Encoder from lz77.py. All tools run with default
parameters.

All sizes are in bytes.

| File                                | raw      |LZ77SlidingWindowEncoder| LZ77Encoder |   gzip   |
|-------------------------------------|----------|------------------------|-------------|----------|
| bootstrap-3.3.6.min.css             |121260    |19394                   |20126        |19747      |
| eff.html                            |41684     |9736                    |10755        |9670       |
| zlib.wasm                           |86408     |38917                   |41156        |37448      |
| jquery-2.1.4.min.js                 |84345     |29521                   |31322        |29569      |
| random.bin (random bytes)           |1000000   |1005470                 |1005050      |1000328    |
| repeat_As.txt                       |1000000   |577                     |578          |1004       |
| kennedy.xls                         |1029744   |207403                  |210182       |204004     |
| alice29.txt                         |152089    |52965                   |54106        |54416      |

"""

import argparse
import tempfile
import unittest
from scl.compressors.lz77 import LZ77Decoder, LZ77Sequence, LZ77StreamsEncoder, LZ77StreamsDecoder
from scl.core.data_block import DataBlock
from scl.core.data_encoder_decoder import DataDecoder, DataEncoder
from scl.core.data_stream import Uint8FileDataStream
from scl.core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
from scl.core.prob_dist import ProbabilityDist
from scl.utils.bitarray_utils import BitArray
from scl.utils.test_utils import (
    create_random_binary_file,
    try_file_lossless_compression,
    try_lossless_compression,
)


class LZ77Window:
    """LZ77 sliding window implementation using a bytearray as the underlying data structure.
    The window is implemented as a circular buffer, so the oldest byte is overwritten when the
    window is full.
    The start and end pointers are used to keep track of the oldest and newest bytes in the window.

    The window is indexed with the absolute position in the data, so the caller need not worry
    about the window being of limited size. An error is thrown if the caller tries to access a
    position beyond the window's end.
    """

    def __init__(self, size, initial_window=None):
        self.data = bytearray(size)
        self.size = size
        self.start = 0
        self.end = 0
        if initial_window is not None:
            for byte in initial_window:
                self.append(byte)

    def append(self, byte):
        if self.end - self.start == self.size:
            self.start += 1
        self.data[self.end % self.size] = byte
        self.end += 1

    def get_byte(self, position):
        if position < self.start or position >= self.end:
            raise IndexError("Position out of range")
        return self.data[position % self.size]

    def get_byte_window_plus_lookahead(self, position, lookahead_buffer):
        """Get byte from the window, or from the lookahead buffer if the position is beyond the window.
        This is useful when we want to assume the lookahead buffer is part of the window, but we don't
        want to actually mutate the window just yet since we might not actually go with this match.
        """
        if position < self.end:
            return self.get_byte(position)
        else:
            return lookahead_buffer[position - self.end]

    def get_window_as_list(self):
        """Return the window as a list from start to end"""
        l = []
        for i in range(self.start, self.end):
            l.append(self.get_byte(i))
        return l


class MatchFinderBase:
    """Base class for LZ77 match finders. The match finder is used to find the best match for a given
    lookahead buffer in the sliding window.
    """

    def reset(self):
        """Reset the match finder. This is useful when we are encoding a new block of data
        that should not depend on the past."""
        pass

    def set_window(self, window):
        """Sets the sliding window to be used by the match finder.
        Specific implementations of the match finder might need to index the window here
        if we are setting a non-empty initial window.
        """
        self.window = window

    def extend_match(
        self, match_start_in_lookahead_buffer, match_pos, lookahead_buffer, left_extension=True
    ):
        """Extend candidate match to the right and left as long as the bytes match.
        Returns the match position and length of the match.

        Parameters:
        lookahead_buffer_pos (int): The index of the candidate start match in the lookahead buffer
        match_pos (int): The position of the candidate start match in the sliding window
        lookahead_buffer (bytearray): The lookahead buffer (this is basically the part not yet
                                      added to window where we are looking for a match).
        left_extension (bool): Whether to try to extend the match to left within the lookahead buffer.

        Returns:
        match_start_in_lookahead_buffer (int): The start index of the best match in the lookahead buffer (might be
                                               different from the candidate start match if we extended left)
        match_pos (int): The position of the best match in the sliding window
        length (int): The length of the best match
        """

        # Right extension:
        # Start comparing bytes from the potential match position in the window and the lookahead buffer
        # until a mismatch is found or we've checked all bytes in the lookahead buffer.
        match_length = 0
        while (
            match_start_in_lookahead_buffer + match_length < len(lookahead_buffer)
            and self.window.get_byte_window_plus_lookahead(
                match_pos + match_length, lookahead_buffer
            )
            == lookahead_buffer[match_start_in_lookahead_buffer + match_length]
        ):
            match_length += 1

        if left_extension:
            # Left extension:
            # If two sequences match at [i, i+length] in the lookahead buffer and [match_pos, match_pos+length] in the window,
            # we then look at the previous byte (i.e., i-1 in the lookahead buffer and match_pos-1 in the window).
            # If they match, we continue moving left until a mismatch is found,
            # ensuring we don't try to extend beyond the window's start or the lookahead buffer's start.

            # Left extension is useful where we missed a hash match because of a collision in the previous
            # step. Note that we only extend left within the lookahead buffer, so we don't tread on the
            # already encoded matches in past.

            # note that this does not change the offset
            while match_pos > self.window.start and (
                match_start_in_lookahead_buffer > 0
                and lookahead_buffer[match_start_in_lookahead_buffer - 1]
                == self.window.get_byte_window_plus_lookahead(match_pos - 1, lookahead_buffer)
            ):
                match_pos -= 1
                match_length += 1
                match_start_in_lookahead_buffer -= 1

        return (
            match_start_in_lookahead_buffer,
            match_pos,
            match_length,
        )  # match_start_in_lookahead_buffer is the number of literals

    def find_best_match(self, lookahead_buffer):
        """
        Find the best match for the current position in the lookahead buffer
        within the sliding window using the hash table.

        Parameters:
        - lookahead_buffer (bytearray): The buffer containing the data that we're trying to find a match for.

        Returns:
        - tuple: A tuple containing the number of literals, the match position, and the match length.
        """
        raise NotImplementedError


class HashBasedMatchFinder(MatchFinderBase):
    """Hash based match finder implementation.
    Uses a hash table of fixed length, stored as a list of lists. Each entry corresponds to
    a hash value and stores a list of previous occurences of that hash value. The list is also
    referred to as a chain below (note that this is different from deflate's or zstd's chain
    hash table since instead of keeping a list of lists they store the older matching positions
    more efficiently in a separate chain array). If more than max_chain_length entries are stored
    in a chain, the oldest entry is removed.

    Parameters:
    - hash_length (int): The length of byte sequences to hash.
    - hash_table_size (int): Size of the hash table.
    - max_chain_length (int): Maximum length of the chains in the hash table.
    - lazy (bool): Whether to use lazy matching where LZ77 considers one step ahead and
                   skips a literal if it finds a longer match.
    - minimum_match_length (int): Minimum length of a match to be considered.
    """

    def __init__(
        self,
        hash_length=4,
        hash_table_size=1000000,
        max_chain_length=64,
        lazy=True,
        minimum_match_length=4,
    ):
        """
        Initializes the chained hash match finder.
        """
        self.hash_length = hash_length
        self.hash_table_size = hash_table_size
        self.hash_table = [[] for _ in range(hash_table_size)]
        self.max_chain_length = max_chain_length
        self.next_position_to_hash = 0
        self.lazy = lazy
        self.minimum_match_length = minimum_match_length

    def reset(self):
        """Reset the match finder. This is useful when we are encoding a new block of data
        that should not depend on the past.
        """
        self.hash_table = [[] for _ in range(self.hash_table_size)]
        self.next_position_to_hash = 0

    def set_window(self, window):
        """Sets the sliding window to be used by the match finder, and indexes the window."""
        super().set_window(window)
        # index the window up to the end of the window minus the hash length
        for j in range(self.window.start, self.window.end - self.hash_length + 1):
            # create the key
            data = bytearray()
            for k in range(j, j + self.hash_length):
                data.append(self.window.get_byte(k))
            self.add_to_hashtable(j, data)

    def _hash(self, data):
        """Computes the hash value of the provided data."""
        # data is a bytearray, which is not hashable
        # so we convert it to a tuple
        return hash(tuple(data)) % len(self.hash_table)

    def get_positions_from_hash(self, data):
        """Retrieve the list of stored positions for the hashed data."""
        h = self._hash(data)
        return self.hash_table[h]

    def add_to_hashtable(self, position, data):
        """Updates the hash table with the new position of a byte sequence. Maintains chain length limit."""
        if len(data) >= self.hash_length:
            h = self._hash(data[: self.hash_length])
            if len(self.hash_table[h]) == self.max_chain_length:
                self.hash_table[h].pop(0)
            self.hash_table[h].append(position)
            self.next_position_to_hash = position + 1

    def find_best_match_at_position(self, lookahead_buffer, i):
        """
        Find the best match for the current position in the lookahead buffer
        within the sliding window using the hash table.

        Parameters:
        - lookahead_buffer (bytearray): The buffer containing the data that we're trying to find a match for.
        - i (int): The position in the lookahead buffer to find a match for.

        Returns:
        - tuple: A tuple containing the number of literals, the match position, and the match length in the best found match.
                 If no match is found, the match length is 0.
        """
        # Retrieve potential match positions from the hash table
        candidate_positions = self.get_positions_from_hash(
            lookahead_buffer[i : i + self.hash_length]
        )

        best_match_pos = 0
        best_length = 0
        best_literals_count = 0

        # Check each potential match position for actual matches
        for match_pos in reversed(candidate_positions):
            offset = self.window.end + i - match_pos
            if offset > self.window.size:
                continue

            literals_count, match_pos, length = self.extend_match(i, match_pos, lookahead_buffer)

            # If a longer match is found
            # (note we go in reverse order so we get the closest match with the longest length)
            if length > best_length:
                best_length = length
                best_match_pos = match_pos
                best_literals_count = literals_count

        return best_literals_count, best_match_pos, best_length

    def find_best_match(self, lookahead_buffer):
        """
        Find the best match for the current position in the lookahead buffer
        within the sliding window using the hash table.

        For the HashBasedMatchFinder this is done by:
        - Iterating over each position `i` in the lookahead buffer.
        - Adding the positions till now to the hash table.
        - Computing a hash for the data at position `i` and calling find_best_match_at_position
          to find the best match for that position.
        - If the match found is less than the minimum match length, we continue to the next `i`, otherwise
          - we keep trying i+1, i+2, etc. until we can find a longer match than the previous step (if lazy=True)
            (this allows us to find longer matches by being not greedy)
        - If no match is found for any i, we return all the positions considered as number of literals with a match_length
          of 0.


        Parameters:
        - lookahead_buffer (bytearray): The buffer containing the data that we're trying to find a match for.

        Returns:
        - tuple: A tuple containing the number of literals, the match position, and the match length.
        """

        i = 0

        # we can look up to the end of the lookahead buffer minus the end part
        # that we can't yet compute the hash for
        num_positions_to_consider = len(lookahead_buffer) - self.hash_length + 1
        if num_positions_to_consider <= 0:
            # this means lookahead_buffer is smaller than the hash length
            # so we just return the length of the buffer as literals
            return (len(lookahead_buffer), 0, 0)

        for i in range(num_positions_to_consider):
            # Ensure all bytes are hashed and added to the hash table
            # We want to index until the end of the window and then also
            # the lookahead buffer until i
            for j in range(max(self.next_position_to_hash, self.window.start), self.window.end + i):
                # create the key
                data = bytearray()
                for k in range(j, j + self.hash_length):
                    data.append(self.window.get_byte_window_plus_lookahead(k, lookahead_buffer))
                self.add_to_hashtable(j, data)

            best_literals_count, best_match_pos, best_length = self.find_best_match_at_position(
                lookahead_buffer, i
            )

            if best_length >= self.minimum_match_length:
                if not self.lazy:
                    return (best_literals_count, best_match_pos, best_length)
                else:
                    # we now keep going forward until the match length keeps increasing
                    # or we reach the end of the lookahead buffer
                    for j in range(i + 1, num_positions_to_consider):
                        (
                            new_best_literals_count,
                            new_best_match_pos,
                            new_best_length,
                        ) = self.find_best_match_at_position(lookahead_buffer, j)
                        if new_best_length > best_length:
                            best_literals_count = new_best_literals_count
                            best_match_pos = new_best_match_pos
                            best_length = new_best_length
                        else:
                            break
                    return (best_literals_count, best_match_pos, best_length)

        return (num_positions_to_consider, 0, 0)


class LZ77SlidingWindowEncoder(DataEncoder):
    """
    LZ77 Sliding Window Encoder

    Parameters:
    match_finder: Match finder used for encoding (not required for decoding)
    window_size: size of sliding window (maximum lookback) - the decoder must use at least
                 this window size (default: 1MB)
    initial_window (optional): initialize window (this is like side information
    or dictionary in zstd parlance). The same initial window should be used for the decoder.

    """

    def __init__(
        self,
        match_finder,
        window_size=1000000,
        initial_window=None,
    ):
        self.match_finder = match_finder
        self.window_size = window_size
        self.window = LZ77Window(window_size, initial_window=initial_window)
        self.match_finder.set_window(self.window)
        self.streams_encoder = LZ77StreamsEncoder()

    def reset(self):
        # reset the window and the match finder
        self.window = LZ77Window(self.window_size)
        self.match_finder.reset()
        self.match_finder.set_window(self.window)

    def lz77_parse_and_generate_sequences(self, data):
        """Encodes the given data using the LZ77 compression algorithm.

        Iterates through the data, finding a match in the sliding window using the matchfinder.
        If a match is found, it outputs a sequence of LZ77 sequences, otherwise it outputs a literal.

        Parameters:
        - data (bytearray): The data to be encoded.

        Returns:
        - lz77_sequences: A list of LZ77 sequences.
        - literals: A bytearray of literals.
        """
        pos_in_data = 0
        lz77_sequences = []
        literals = bytearray()

        while pos_in_data < len(data):
            lookahead = data[pos_in_data:]

            # try to find a match
            literals_count, match_pos, match_length = self.match_finder.find_best_match(lookahead)
            if match_length > 0:
                # match found

                # compute offset
                match_offset = self.window.end + literals_count - match_pos

                # append the literals before the match starts
                literals.extend(data[pos_in_data : pos_in_data + literals_count])
                # append the LZ77 sequence
                lz77_sequences.append(LZ77Sequence(literals_count, match_length, match_offset))

                bytes_done = literals_count + match_length

                # insert the parsed data into the window for future matches
                for j in range(bytes_done):
                    self.window.append(data[pos_in_data + j])
                pos_in_data += bytes_done
            else:
                # no match found, insert the data into literals and into the window
                for j in range(literals_count):
                    literals.append(data[pos_in_data + j])
                    self.window.append(data[pos_in_data + j])
                pos_in_data += literals_count
                # append an sequence with just literals
                lz77_sequences.append(LZ77Sequence(literals_count, 0, 0))

        return lz77_sequences, literals

    def encode_block(self, data_block: DataBlock):
        # first do lz77 parsing
        lz77_sequences, literals = self.lz77_parse_and_generate_sequences(data_block.data_list)
        # now encode sequences and literals
        encoded_bitarray = self.streams_encoder.encode_block(lz77_sequences, literals)
        return encoded_bitarray

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int = 10000):
        """utility wrapper around the encode function using Uint8FileDataStream

        Args:
            input_file_path (str): path of the input file
            encoded_file_path (str): path of the encoded binary file
            block_size (int): choose the block size to be used to call the encode function
        """
        # call the encode function and write to the binary file
        with Uint8FileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class LZ77SlidingWindowDecoder(DataDecoder):
    """
    LZ77 Sliding Window Decoder

    Parameters:
    window_size: size of sliding window (maximum lookback) - the decoder must use at least
                 the window size used by the encoder (default: 1MB)
    initial_window (optional): initialize window (this is like side information
    or dictionary in zstd parlance). The same initial window should be used for the decoder.

    """

    def __init__(self, window_size=1000000, initial_window=None):
        self.window = LZ77Window(window_size, initial_window=initial_window)

        self.streams_decoder = LZ77StreamsDecoder()

    def execute_lz77_sequences(self, literals, lz77_sequences):
        """Executes the LZ77 sequences and the literals and returns the decoded bytes.
        Execution here just means the decoding.

        Updates the window accordingly.
        """
        decoded = bytearray()
        pos_in_literals = 0

        for sequence in lz77_sequences:
            # Copy literals directly to the output
            for _ in range(sequence.literal_count):
                byte = literals[pos_in_literals]
                pos_in_literals += 1
                decoded.append(byte)
                self.window.append(byte)

            # Copy from the offset to the output
            for _ in range(sequence.match_length):
                byte = self.window.get_byte(self.window.end - sequence.match_offset)
                decoded.append(byte)
                self.window.append(byte)

        # Copy any remaining literals to the output at the end of block
        for i in range(pos_in_literals, len(literals)):
            decoded.append(literals[i])
            self.window.append(literals[i])
        return decoded

    def decode_block(self, encoded_bitarray: BitArray):
        # first entropy decode the lz77 sequences and the literals
        (lz77_sequences, literals), num_bits_consumed = self.streams_decoder.decode_block(
            encoded_bitarray
        )

        # now execute the sequences to decode
        decoded_block = DataBlock(self.execute_lz77_sequences(literals, lz77_sequences))
        return decoded_block, num_bits_consumed

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function using Uint8FileDataStream

        Args:
            encoded_file_path (str): input binary file
            output_file_path (str): output (text) file to which decoded data is written
        """

        # decode data and write output to a text file
        with EncodedBlockReader(encoded_file_path) as reader:
            with Uint8FileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


class LZ77WindowTest(unittest.TestCase):
    def test_LZ77Window(self):
        window = LZ77Window(4)

        # Appending bytes
        window.append(65)  # 'A'
        window.append(66)  # 'B'
        window.append(67)  # 'C'
        assert window.data == bytearray([65, 66, 67, 0])
        assert window.get_byte(1) == 66

        # Overflow the window
        window.append(68)  # 'D'
        window.append(69)  # 'E' - This will overwrite 'A'
        assert window.data == bytearray([69, 66, 67, 68])
        assert window.get_byte(4) == 69

    @unittest.expectedFailure
    def test_LZ77Window_get_byte_out_of_range_too_big(self):
        window = LZ77Window(4)
        window.append(65)
        window.append(66)
        window.append(67)
        window.append(68)
        window.append(69)
        window.get_byte(5)  # This should raise an IndexError

    @unittest.expectedFailure
    def test_LZ77Window_get_byte_out_of_range_too_small(self):
        window = LZ77Window(4)
        window.append(65)
        window.append(66)
        window.append(67)
        window.append(68)
        window.append(69)
        window.get_byte(0)  # This should raise an IndexError


import os


def test_lz77_encode_decode():
    for data in [
        b"ABABABABABCDABABABABABCDEDEDEDEDE",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        b"This is a simple test. This is a simple test.",
        b"Hello World! " * 10,
        os.urandom(100),
        b"A" * 100,
    ]:
        for window_size in [1, 16, 4096]:
            for initial_window in [None, data[:10]]:
                for hash_length in [2, 3, 4]:
                    for hash_table_size in [1, 4096]:
                        for max_chain_length in [1, 32]:
                            for lazy in [False, True]:
                                for minimum_match_length in [1, 4]:
                                    match_finder = HashBasedMatchFinder(
                                        hash_length=hash_length,
                                        hash_table_size=hash_table_size,
                                        max_chain_length=max_chain_length,
                                        lazy=lazy,
                                        minimum_match_length=minimum_match_length,
                                    )
                                    encoder = LZ77SlidingWindowEncoder(
                                        match_finder,
                                        window_size=window_size,
                                        initial_window=initial_window,
                                    )
                                    decoder = LZ77SlidingWindowDecoder(
                                        window_size=window_size, initial_window=initial_window
                                    )

                                    is_lossless, _, _ = try_lossless_compression(
                                        DataBlock(data),
                                        encoder,
                                        decoder,
                                        add_extra_bits_to_encoder_output=True,
                                    )
                                    assert is_lossless

                                    # try giving a longer window to the decoder and make sure it still works
                                    match_finder = HashBasedMatchFinder(
                                        hash_length=hash_length,
                                        hash_table_size=hash_table_size,
                                        max_chain_length=max_chain_length,
                                        lazy=lazy,
                                    )
                                    encoder = LZ77SlidingWindowEncoder(
                                        match_finder,
                                        window_size=window_size,
                                        initial_window=initial_window,
                                    )
                                    decoder = LZ77SlidingWindowDecoder(
                                        window_size=window_size + 100, initial_window=initial_window
                                    )
                                    is_lossless, _, _ = try_lossless_compression(
                                        DataBlock(data),
                                        encoder,
                                        decoder,
                                        add_extra_bits_to_encoder_output=True,
                                    )
                                    assert is_lossless

                                    # now test that we can also decode using lz77.py
                                    match_finder = HashBasedMatchFinder(
                                        hash_length=hash_length,
                                        hash_table_size=hash_table_size,
                                        max_chain_length=max_chain_length,
                                        lazy=lazy,
                                    )
                                    encoder = LZ77SlidingWindowEncoder(
                                        match_finder,
                                        window_size=window_size,
                                        initial_window=initial_window,
                                    )
                                    non_sliding_window_decoder = LZ77Decoder(
                                        initial_window=initial_window
                                    )
                                    is_lossless, _, _ = try_lossless_compression(
                                        DataBlock(data),
                                        encoder,
                                        non_sliding_window_decoder,
                                        add_extra_bits_to_encoder_output=True,
                                    )


def test_lz77_sequence_generation():
    """
    Test that lz77 produces expected sequences (based on similar test in lz77.py).
    Use a big enough window so that we don't have to worry about the window being full.
    Also test behavior across blocks both when we reset and when we don't.
    """
    min_match_len = 3
    initial_window = [0, 0, 1, 1, 1]
    match_finder = HashBasedMatchFinder(
        hash_length=3,
        hash_table_size=100000,
        max_chain_length=64,
        lazy=False,
        minimum_match_length=3,
    )
    encoder = LZ77SlidingWindowEncoder(match_finder, initial_window=initial_window)
    decoder = LZ77SlidingWindowDecoder(initial_window=initial_window)

    data_list = [
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        255,
        254,
        255,
        254,
        255,
        254,
        255,
        2,
        0,
        0,
        1,
        1,
        1,
        1,
        44,
    ]

    # matches here are (first one and third one are overlapping, last one picks longer match which is not the most recent match for 3-tuple):
    # 0, 0, 1, 1, [1, [1, 1, 1,] 1] 0, 0, 1, 1, 1, 255, 254, 255, 254, 255, 254, 255, 2, 0, 0, 1, 1, 1, 1, 44
    # [0, 0, 1, 1, 1,] 1, 1, 1, [0, 0, 1, 1, 1,] 255, 254, 255, 254, 255, 254, 255, 2, 0, 0, 1, 1, 1, 1, 44
    # 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, [255, 254, [255, 254, 255,] 254, 255,] 2, 0, 0, 1, 1, 1, 1, 44
    # [0, 0, 1, 1, 1, 1,] 1, 1, 0, 0, 1, 1, 1, 255, 254, 255, 254, 255, 254, 255, 2, [0, 0, 1, 1, 1, 1], 44

    expected_lits = [255, 254, 2, 44]
    expected_seqs = [
        LZ77Sequence(0, 4, 1),
        LZ77Sequence(0, 5, 9),
        LZ77Sequence(2, 5, 2),
        LZ77Sequence(1, 6, 22),
        LZ77Sequence(1, 0, 0),
    ]
    seqs, lits = encoder.lz77_parse_and_generate_sequences(data_list)

    window_data = encoder.window.get_window_as_list()
    assert window_data == initial_window + data_list
    assert (
        sum(len(v) for v in encoder.match_finder.hash_table) == len(data_list) - min_match_len + 1
    )  # subtract 2 since min_match_len is 3
    assert list(lits) == expected_lits
    assert seqs == expected_seqs

    # try to decode and verify
    decoded = decoder.execute_lz77_sequences(lits, seqs)
    assert list(decoded) == data_list

    # encode another block which is copy of first and see that we get just one match
    seqs, lits = encoder.lz77_parse_and_generate_sequences(data_list)
    window_data = encoder.window.get_window_as_list()
    assert window_data == initial_window + data_list * 2

    # Remove below check since the indexing is lazy and it won't index the whole thing just yet
    # since it found a long match

    # assert (
    #     sum(len(v) for v in encoder.match_finder.hash_table) == len(data_list) - min_match_len + 1
    # ) # subtract 2 since min_match_len is 3

    assert list(lits) == []
    assert seqs == [LZ77Sequence(0, len(data_list), len(data_list))]

    # try to decode and verify
    decoded = decoder.execute_lz77_sequences(lits, seqs)
    assert list(decoded) == data_list

    # now reset encoder and verify that after encoding we get results that we expect without initial window

    # matches:
    # [1, [1, 1,] 1,] 0, 0, 1, 1, 1, 255, 254, 255, 254, 255, 254, 255, 2, 0, 0, 1, 1, 1, 1, 44
    # 1, [1, 1, 1,] 0, 0, [1, 1, 1,] 255, 254, 255, 254, 255, 254, 255, 2, 0, 0, 1, 1, 1, 1, 44
    # 1, 1, 1, 1, 0, 0, 1, 1, 1, [255, 254, [255, 254, 255,] 254, 255,] 2, 0, 0, 1, 1, 1, 1, 44
    # 1, 1, 1, 1, [0, 0, 1, 1, 1,] 255, 254, 255, 254, 255, 254, 255, 2, [0, 0, 1, 1, 1,] 1, 44
    encoder.reset()
    decoder = LZ77SlidingWindowDecoder()
    expected_lits = [1, 0, 0, 255, 254, 2, 1, 44]
    expected_seqs = [
        LZ77Sequence(1, 3, 1),
        LZ77Sequence(2, 3, 5),
        LZ77Sequence(2, 5, 2),
        LZ77Sequence(1, 5, 13),
        LZ77Sequence(2, 0, 0),
    ]
    seqs, lits = encoder.lz77_parse_and_generate_sequences(data_list)
    window_data = encoder.window.get_window_as_list()
    assert window_data == data_list

    # removing test below since indexing is lazy and it won't index the whole thing just yet
    # since it's only left with 2 bytes at the end which is less than the hash length
    # so it doesn't index the last match also
    # assert (
    #     sum(len(v) for v in encoder.match_finder.hash_table) == len(data_list) - min_match_len + 1
    # )
    assert list(lits) == expected_lits
    assert seqs == expected_seqs

    # try to decode and verify
    decoded = decoder.execute_lz77_sequences(lits, seqs)
    assert list(decoded) == data_list


class LZ77DecoderWindowTooSmallTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_LZ77DecoderWindowTooSmall(self):
        data = b"ABABABABABCDABABABABABCDEDEDEDEDE"
        match_finder = HashBasedMatchFinder()
        encoder = LZ77SlidingWindowEncoder(match_finder, window_size=16)
        decoder = LZ77SlidingWindowDecoder(window_size=8)
        encoded_data = encoder.encode_block(DataBlock(data))
        decoded = decoder.decode_block(encoded_data)


def test_lz77_multiblock_file_encode_decode():
    """full test for LZ77SlidingWindowEncoder and LZ77SlidingWindowDecoder

    - create a sample file
    - encode the file using LZ77SlidingWindowEncoder
    - perform decoding and check if the compression was lossless

    """
    initial_window = [44, 45, 46] * 5
    # define encoder, decoder
    match_finder = HashBasedMatchFinder()
    encoder = LZ77SlidingWindowEncoder(
        match_finder, window_size=1100, initial_window=initial_window
    )
    decoder = LZ77SlidingWindowDecoder(window_size=1100, initial_window=initial_window)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # create a file with some random data
        input_file_path = os.path.join(tmpdirname, "inp_file.txt")
        create_random_binary_file(
            input_file_path,
            file_size=5000,
            prob_dist=ProbabilityDist({44: 0.5, 45: 0.25, 46: 0.2, 255: 0.05}),
        )

        # test lossless compression
        assert try_file_lossless_compression(
            input_file_path, encoder, decoder, encode_block_size=1000
        )


if __name__ == "__main__":
    # Provide a simple CLI interface below for convenient experimentation
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--decompress", help="decompress", action="store_true")
    parser.add_argument("-i", "--input", help="input file", required=True, type=str)
    parser.add_argument("-o", "--output", help="output file", required=True, type=str)
    parser.add_argument(
        "-w", "--window_init", help="initialize window from file (like zstd dictionary)", type=str
    )

    # constants
    BLOCKSIZE = 100_000  # encode in 100 KB blocks

    args = parser.parse_args()

    initial_window = None
    if args.window_init is not None:
        with open(args.window_init, "rb") as f:
            initial_window = list(f.read())

    if args.decompress:
        decoder = LZ77SlidingWindowDecoder(initial_window=initial_window)
        decoder.decode_file(args.input, args.output)
    else:
        match_finder = HashBasedMatchFinder()
        encoder = LZ77SlidingWindowEncoder(match_finder, initial_window=initial_window)
        encoder.encode_file(args.input, args.output, block_size=BLOCKSIZE)
