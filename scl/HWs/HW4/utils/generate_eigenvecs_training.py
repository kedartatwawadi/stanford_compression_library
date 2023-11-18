import numpy as np
import os

def generate_eigenvecs_training_data(data_npy, block_size):
    data_vectors = data_npy.reshape(-1, block_size)
    # get the covariance matrix
    cov = np.cov(data_vectors, rowvar=False)
    # get the eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    return cov, eigenvals, eigenvecs


def get_eigendata(block_size, rho, sigma, root_path=''):
    file_name = f'cov_rho{rho}_sigma{sigma}_blocksize{block_size}.npy'
    cov = np.load(os.path.join(root_path, file_name))

    file_name = f'eigenvals_rho{rho}_sigma{sigma}_blocksize{block_size}.npy'
    eigenvals = np.load(os.path.join(root_path, file_name))

    file_name = f'eigenvecs_rho{rho}_sigma{sigma}_blocksize{block_size}.npy'
    eigenvecs = np.load(os.path.join(root_path, file_name))

    return cov, eigenvals, eigenvecs


def main():
    # data parameters
    num_samples = 100_000
    sigma = 1
    sigma_0 = sigma
    rho_set = [0.5, 0.9, 0.99]

    # codebook parameters
    block_sizes = [2, 4]

    for rho in rho_set:
        # load training data
        training_data = np.load(f'../p4_v2_data/training_data_rho{rho}_sigma{sigma}.npy')

        # generate codebook
        for block_size in block_sizes:
            print(f"Generating eigenvecs for rho = {rho}, block_size = {block_size}")
            cov, eigenvals, eigenvecs = generate_eigenvecs_training_data(training_data, block_size)
            np.save(f'../p5_data/'
                    f'cov_rho{rho}_sigma{sigma}_blocksize{block_size}.npy', cov)
            np.save(f'../p5_data/'
                    f'eigenvals_rho{rho}_sigma{sigma}_blocksize{block_size}.npy', eigenvals)
            np.save(f'../p5_data/'
                    f'eigenvecs_rho{rho}_sigma{sigma}_blocksize{block_size}.npy', eigenvecs)


if __name__ == '__main__':
    main()