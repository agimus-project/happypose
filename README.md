# HappyPose

[![Conda](https://github.com/agimus-project/happypose/actions/workflows/conda.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/conda.yml)
[![Pip](https://github.com/agimus-project/happypose/actions/workflows/pip.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/pip.yml)
[![Poetry](https://github.com/agimus-project/happypose/actions/workflows/poetry.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/poetry.yml)
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

### With conda

```
git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
cd happypose
conda env create -f environment.yml
conda activate happypose
pip install .[cpu,evaluation,multiview,render]
```

### With venv

```
git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
cd happypose
python -m venv .venv
source .venv/bin/activate
pip install .[cpu,evaluation,multiview,render]
```

## Create data directory

```
Create data dir /somewhere/convenient. The dataset to store are quite large.
export HAPPYPOSE_DATA_DIR=/somewhere/convenient
```
