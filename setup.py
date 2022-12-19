from setuptools import setup

setup(
    name="stanford_compression_library",
    version="0.0.1",
    author="Kedar Tatwawadi, Shubham Chandak, Pulkit Tandon",
    author_email="ktatwawadi@gmail.com",
    packages=["scl"],
    description="A library of common compressors",
    license="MIT",
    install_requires=[
        "pytest",
        "numpy",
        "bitarray",
        "zstandard",
    ],
)
