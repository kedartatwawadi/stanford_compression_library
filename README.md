# Stanford Compression Library
The goal of the library is to help with research in the area of data compression. This is not meant to be fast or efficient implementation, but rather for educational purpose

## Getting started
- Create a virtualenv [TODO] (tested with python ver:`3.8.2`)
- Install required packages [TODO]
```
pip install -r requirements.txt
```
- Add path to the repo to `PYTHONPATH`
For example:
```
export PYTHONPATH=$PYTHONPATH:/Users/kedar/code/stanford_compression_library
```
- Run unit tests

To run all tests:
```
find . -name "*_tests.py" -exec py.test -s -v {} +
```

To run a single test
```
py.test -s -v core/data_stream_tests.py
```

## Getting started with understanding the library
In-depth information about the library will be in the comments. Tutorials/articles etc will be posted on the wiki page: 
https://github.com/kedartatwawadi/stanford_compression_library/wiki/Introduction-to-the-Stanford-Compression-Library

## How to submit code

Run a formatter before submitting PR
```
black <dir/file> --line-length 100
```

## Contact
The best way to contact the maintainers is to file an issue with your question. 
If not please use the following email:
- Kedar Tatwawadi: kedart@stanford.edu
