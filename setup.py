from setuptools import setup, find_packages

version = "0.0.1"

setup(
    name="pygho",
    version=version,
    description="PygHO is a library for high-order GNNs",
    download_url="https://github.com/GraphPKU/PygHO",
    author="GraphPKU",
    python_requires=">=3.10",
    packages=find_packages(include=["pygho", "pygho.*"]),
    install_requires=[
        "torch",
        "torch_scatter",
        "torch_geometric"
    ],
)