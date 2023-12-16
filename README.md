# Stanford Compression Library
The goal of the library is to help with research in the area of data compression. This is not meant to be fast or efficient implementation, but rather for educational purpose

This library is currently being used for the course [EE274: Data Compression course, Fall 22](https://stanforddatacompressionclass.github.io/Fall22/) at Stanford University, to augment the lectures and for homeworks: 
1. [EE274 Lecture materials (slides etc)](https://stanforddatacompressionclass.github.io/Fall22/lectures/)
2. [EE274 Course notes (in progress)](https://stanforddatacompressionclass.github.io/notes/)
3. The video recordings for lectures can be found as a [Youtube Playlist](https://youtube.com/playlist?list=PLv_7iO_xlL0Jgc35Pqn7XP5VTQ5krLMOl)


## Compression algorithms
Here is a list of algorithms implemented.

### Prefix-free codes
- [Huffman codes](scl/compressors/huffman_coder.py)
- [Shannon codes](scl/compressors/shannon_coder.py)
- [Fano codes](scl/compressors/fano_coder.py)
- [Shannon Fano Elias](scl/compressors/shannon_fano_elias_coder.py)
- [Golomb codes](scl/compressors/golomb_coder.py)
- [Universal integer coder](scl/compressors/universal_uint_coder.py)
- [Elias Delta code](scl/compressors/elias_delta_uint_coder.py)

### Entropy coders
- [rANS](scl/compressors/rANS.py)
- [tANS](scl/compressors/tANS.py)
- [Typical set coder](scl/compressors/typical_set_coder.py)
- [Arithmetic coder](scl/compressors/arithmetic_coding.py)
- [Context-based Adaptive Arithmetic coder](scl/compressors/probability_models.py)
- [Range coder](scl/compressors/range_coder.py)

### Universal lossless compressors
- [zlib (external)](scl/external_compressors/zlib_external.py)
- [zstd (external)](scl/external_compressors/zstd_external.py)
- [LZ77](scl/compressors/lz77.py)
- [LZ77 (sliding window version)](scl/compressors/lz77_sliding_window.py)


NOTE -> the tests in each file should be helpful as a "usage" example of each of the compressors. More details are on the project wiki.


## Getting started
- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8
    conda activate myenv
    ```
- Clone the repo
    ```
    git clone https://github.com/kedartatwawadi/stanford_compression_library.git
    cd stanford_compression_library
    ```
- Install the `scl` package
    ```
    pip install -e . #install the package in a editable mode
    ``` 

- **Run unit tests**

  To run all tests:
    ```
    find scl -name "*.py" -exec py.test -s -v {} +
    ```

  To run a single test
  ```
  py.test -s -v scl/core/data_stream.py
  ```

## Getting started with understanding the library
In-depth information about the library will be in the comments. Tutorials/articles etc will be posted on the wiki page: 
https://github.com/kedartatwawadi/stanford_compression_library/wiki/Introduction-to-the-Stanford-Compression-Library

## How to submit code

Run a formatter before submitting PR
```
black <dir/file> --line-length 100
```

Note that the Github actions CI uses black, so the PR will fail if black is not used. (see [`.github/workflows/black.yml`](.github/workflows/black.yml)),  
#
# Contact
The best way to contact the maintainers is to file an issue with your question. 
