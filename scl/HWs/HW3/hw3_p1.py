import argparse
import os
from scl.compressors.lz77 import LZ77Encoder
from scl.core.data_block import DataBlock

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_dir", help="input directory", required=True, type=str
    )
    parser.add_argument(
        "-s", "--seed_input_file", help="initialize window from seed input file", type=str
    )
    args = parser.parse_args()

    window_initialization = None
    if args.seed_input_file is not None:
        print(f"Loading seed input from {args.seed_input_file}")
        with open(args.seed_input_file, "rb") as f:
            window_initialization = list(f.read())
    else:
        print("Compressing without using a seed input")

    total_size_individually_compressed = 0
    num_files = 0
    total_uncompressed_size = 0
    concatenated_data = []

    # loop over files in directory and individually compress
    # also concatenate data into concatenated_data
    for filename in os.listdir(args.input_dir):
        filename = os.path.join(args.input_dir, filename)
        with open(filename, "rb") as f:
            data = list(f.read())
        total_uncompressed_size += len(data)
        num_files += 1
        encoder = LZ77Encoder(initial_window=window_initialization)
        encoded = encoder.encode_block(DataBlock(data))
        total_size_individually_compressed += len(encoded)
        concatenated_data += data

    # joint compression
    encoder = LZ77Encoder(initial_window=window_initialization)
    encoded = encoder.encode_block(DataBlock(concatenated_data))
    total_size_jointly_compressed = len(encoded)
    print("Number of files:", num_files)
    print("Total uncompressed size (in bits):", total_uncompressed_size*8)
    print("Normalized uncompressed size (in avg. bits/file):", total_uncompressed_size//num_files*8)
    print(
        "Total size after compressing the files individually (in bits):",
        total_size_individually_compressed,
    )
    print(
        "Total size after compressing the files jointly (in bits):",
        total_size_jointly_compressed,
    )
