import numpy as np
import json
import os
from lossy_compressors.vector_quantizer import (
    build_kmeans_codebook_fast,
)

def generate_samples(num_samples, rho, sigma=1, sigma_0=1):
    """
    num_samples -> number of samples to generate
    rho -> correlation between two consecutive samples
    sigma -> std of the gaussian

    returns -> a list of samples from a gaussian markov process such that
    x[n] = rho * x[n-1] + sqrt(1 - rho^2) * z[n]
    where z[n] is a gaussian random variable with mean 0 and std sigma
    x[0] is a gaussian random variable with mean 0 and std sigma_0
    """
    samples = np.zeros(num_samples)
    samples[0] = np.random.normal(0, sigma_0)
    for i in range(1, num_samples):
        samples[i] = rho * samples[i - 1] + np.sqrt((1 - rho**2)) * np.random.normal(0, sigma)
    return samples


def generate_VQ_codebook(training_data, block_size, num_bits_per_symbol):
    """
    training_data -> a numpy array of training data
    block_size -> the size of the blocks to split the training data into
    num_bits_per_symbol -> the number of bits per symbol

    returns -> codebook of shape (num_vectors, block_size) where num_vectors depends on num_bits_per_symbol
    """
    # calculate number of vectors in the codebook
    num_vector_bits = num_bits_per_symbol * block_size
    num_vectors = int(2 ** num_vector_bits)

    # break data into blocks
    data_block = np.reshape(training_data, [-1, block_size])

    # generate codebook
    codebook = build_kmeans_codebook_fast(data_block, num_vectors)

    assert codebook.shape == (num_vectors, block_size)

    return codebook


def get_codebook(block_size, num_bits_per_symbol, rho, sigma, root_path=''):
    file_name = f'codebook_rho{rho}_sigma{sigma}_blocksize{block_size}_bps{float(num_bits_per_symbol)}.npy'
    return np.load(os.path.join(root_path, file_name))


def main():
    # data parameters
    num_samples = 100_000
    sigma = 1
    sigma_0 = sigma
    rho_set = [0.5, 0.9, 0.99]

    # codebook parameters
    block_sizes = [1, 2, 4, 8, 16]
    total_num_bits_per_symbol = 1

    for rho in rho_set:
        # generate training data
        print(f"Generating training data for rho = {rho}")
        training_data = generate_samples(num_samples, rho, sigma)
        np.save(f'../p4_v2_data/training_data_rho{rho}_sigma{sigma}.npy', training_data)

        # generate codebook
        for block_size in block_sizes:
            step_size = total_num_bits_per_symbol/block_size
            if block_size <= 8:
                num_bits_per_block_sweep = np.arange(step_size, total_num_bits_per_symbol + step_size, step_size)
            else:
                num_bits_per_block_sweep = np.arange(step_size, total_num_bits_per_symbol/2 + step_size, step_size)

            for num_bits_per_block in total_num_bits_per_symbol:
                print(f"Generating codebook for rho = {rho}, block_size = {block_size}, "
                      f"num_bits_per_symbol = {num_bits_per_block}")
                codebook = generate_VQ_codebook(training_data, block_size, num_bits_per_block)
                np.save(f'../p4_v2_data/codebooks/'
                        f'codebook_rho{rho}_sigma{sigma}_blocksize{block_size}_bps{num_bits_per_block}.npy', codebook)


if __name__ == '__main__':
    main()