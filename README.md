# Stanford Compression Library
The goal of the library is to help with research in the area of data compression. This is not meant to be fast or efficient implementation, but rather for educational purpose

This library is currently being used for the course [EE274: Data Compression course, Fall 22](https://stanforddatacompressionclass.github.io/Fall22/) at Stanford University, to augment the lectures and for homeworks: 
1. [EE274 Lecture materials (slides etc)](https://stanforddatacompressionclass.github.io/Fall22/lectures/)
2. [EE274 Course notes (in progress)](https://stanforddatacompressionclass.github.io/notes/)
3. The video recordings for lectures can be found as a [Youtube Playlist](https://youtube.com/playlist?list=PLv_7iO_xlL0Jgc35Pqn7XP5VTQ5krLMOl)


## Compression algorithms
Here is a list of algorithms implemented.

### Prefix-free codes
- [Huffman codes](compressors/huffman_coder.py)
- [Shannon codes](compressors/shannon_coder.py)
- [Fano codes](compressors/fano_coder.py)
- [Shannon Fano Elias](compressors/shannon_fano_elias_coder.py)
- [Golomb codes](compressors/golomb_coder.py)
- [Universal integer coder](compressors/universal_integer_coder.py)
- [Elias Delta code](compressors/elias_delta_uint_coder.py)

### Entropy coders
- [rANS](compressors/rANS.py)
- [tANS](compressors/tANS.py)
- [Typical set coder](compressors/typical_set_coder.py)
- [Arithmetic coder](compressors/arithmetic_coding.py)
- [Context-based Adaptive Arithmetic coder](compressors/probability_models.py)
- [Range coder](compressors/range_coder.py)

### Universal Lossless compressors
- [zlib (external)](external_compressors/zlib_external.py)
- [zstd (external)](external_compressors/zstd_external.py)
- [LZ77](compressors/lz77.py)


NOTE -> the tests in each file should be helpful as a "usage" example of each of the compressors. More details are on the project wiki.


## Getting started
- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8
    conda activate myenv
    
    ```
- Install the `scl` package
    ```
    pip install -e . #install the package in a editable mode
    ``` 

- **Run unit tests**

  To run all tests:
    ```
    find  scl -name "*.py" -exec py.test -s -v {} +
    ```

  To run a single test
  ```
  py.test -s -v scl/core/data_stream_tests.py
  ```

## Getting started with understanding the library
In-depth information about the library will be in the comments. Tutorials/articles etc will be posted on the wiki page: 
https://github.com/kedartatwawadi/stanford_compression_library/wiki/Introduction-to-the-Stanford-Compression-Library

## How to submit code

Run a formatter before submitting PR
```
black <dir/file> --line-length 100
```

Note that the Github actions CI uses flake8 as a lint (see [`.github/workflows/python-app.yml`](.github/workflows/python-app.yml)), which is compatible with the `black` formatter as discussed [here](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html#flake8).

## Contact
The best way to contact the maintainers is to file an issue with your question. 
