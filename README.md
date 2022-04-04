# Stanford Compression Library
The goal of the library is to help with research in the area of data compression. This is not meant to be fast or efficient implementation, but rather for educational purpose

## Getting started
- Create conda environment and install required packages:
    ```
    conda create --name myenv python=3.8.2
    conda activate myenv
    python -m pip install -r requirements.txt
    ```
- Add path to the repo to `PYTHONPATH`:
    ```
    export PYTHONPATH=$PYTHONPATH:$(pwd)
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


## How to submit code

Run a formatter before submitting PR
```
black <dir/file> --line-length 100
```

Note that the Github actions CI uses flake8 as a lint (see [`.github/workflows/python-app.yml`](.github/workflows/python-app.yml)), which is compatible with the `black` formatter as discussed [here](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html#flake8).

## Contact
The best way to contact the maintainers is to file an issue with your question. 
If not please use the following email:
- Kedar Tatwawadi: kedart@stanford.edu
