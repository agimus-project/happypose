# HappyPose

[![Conda](https://github.com/agimus-project/happypose/actions/workflows/conda-test.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/conda-test.yml)
[![Pip](https://github.com/agimus-project/happypose/actions/workflows/pip-test.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/pip-test.yml)
[![Poetry](https://github.com/agimus-project/happypose/actions/workflows/poetry-test.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/poetry-test.yml)
[![Book](https://github.com/agimus-project/happypose/actions/workflows/book.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/book.yml)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/agimus-project/happypose/main.svg)](https://results.pre-commit.ci/latest/github/agimus-project/happypose/main)
[![Documentation Status](https://readthedocs.org/projects/happypose/badge/?version=latest)](https://happypose.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/agimus-project/happypose/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/agimus-project/happypose)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Toolbox and trackers for object pose-estimation. Based on the work [CosyPose](https://github.com/Simple-Robotics/cosypose) and [MegaPose](https://github.com/megapose6d/megapose6d). This directory is currently under development. Please refer to the [documentation](https://agimus-project.github.io/happypose/) for more details.


## Installation

To install happypose, you can use pip or poetry.

We strongly suggest to install it in either a
[venv](https://docs.python.org/fr/3/library/venv.html) or a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Example with conda

```
git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
cd happypose
conda env create -f environment.yml
conda activate happypose
pip install .
```

### Example with venv

```
git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
cd happypose
python -m venv .venv
source .venv/bin/activate
pip install .[pypi,cpu] --extra-index-url https://download.pytorch.org/whl/cpu
```

### Install extras:

- `cpu`: required to get pytorch CPU from PyPI (don't use this for GPU or with conda)
- `gpu`: required to get pytorch GPU from PyPI (don't use this for CPU or with conda)
- `multiview`: installs cosypose c++ extension
- `pypi`: install pinocchio & opencv from PyPI (don't use this with conda)

## Create data directory

```
Create data dir /somewhere/convenient. The dataset to store are quite large.
export HAPPYPOSE_DATA_DIR=/somewhere/convenient
```

## Test the install

### CPU

If you work on CPU, these models need to be download :

```
#hope dataset models for CosyPose
python -m happypose.toolbox.utils.download --cosypose_models \
            detector-bop-hope-pbr--15246 \
            coarse-bop-hope-pbr--225203 \
            refiner-bop-hope-pbr--955392
```

```
# For MegaPose
python -m happypose.toolbox.utils.download --megapose_models
```

and the examples

```
python -m happypose.toolbox.utils.download --examples barbecue-sauce
```

In the HappyPose folder:

```
pytest -v ./tests
```

You may need to install `pytest-order` : `pip installp pytest-order`. In this case, test related to the `evaluation` and the `training` of CosyPose are not run. If you want to use these functionalities, you need a GPU.

### GPU

Tests related to `evaluation` and `training` will be run if a GPU is available. Hence, a few more downloads are needed :

```
#ycbv models
python -m happypose.toolbox.utils.download --cosypose_models \
            coarse-bop-ycbv-pbr--724183 \
            refiner-bop-ycbv-pbr--604090
```

```
python -m happypose.toolbox.utils.download --bop_dataset ycbv
```

```
python -m happypose.toolbox.utils.download --test-results
```

The tests take much longer in this case.