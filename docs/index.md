# Getting Started

**Welcome to the documentation for NEMO initial conditions (pyIC)**

## Introduction

______________________________________________________________________

## Configuration :globe_with_meridians:

______________________________________________________________________

## Quick Start :rocket:

### Installation

To get started, check out and set up an instance of the pyIC GitHub [repository](https://github.com/NOC-MSM/pyIC):

```sh
export PYIC_DIR=$PWD/pyIC
git clone git@github.com:NOC-MSM/pyIC.git
```

??? tip "Helpful Tip..."

```
* **It is not advised to checkout the respository in your home directory.**
```

Create a specific conda virtual environment. Load conda (e.g. through anaconda/miniforge) and create the environment through the provided `environment.yml` file.

```sh
cd $PYIC_DIR
conda env create -n pyic -f environment.yml
```

Activate the new environment

```sh
conda activate pyic
```

Install pyIC

```sh
pip install -e .
```

### Usage

pyIC revolves around its `GRID` class, which takes a gridded data set as input (such as a `netCDF`, or any other file that can be opened using `xarray`). Then we use the Regridder within `xesmf` to regrid data on one `GRID` to another.

A basic example is included within the `pyic_exe.py` script and can be run from the command line. Further arguments are required to specify x, y, depth if they are not on a list of commonly inferred ones.

```sh
python pyic_exe.py file1.nc file2.nc
```

### Example scripts

Several example python scripts can be found in the `examples/` subdirectory.
These are split into two rough categories: synthetic data (with scripts to create said data) and data from NEMO and other models.
