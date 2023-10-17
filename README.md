# HappyPose

[![Tests](https://github.com/agimus-project/happypose/actions/workflows/test.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/test.yml)
[![Packaging](https://github.com/agimus-project/happypose/actions/workflows/packaging.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/packaging.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/agimus-project/happypose/main.svg)](https://results.pre-commit.ci/latest/github/agimus-project/happypose/main)
[![Documentation Status](https://readthedocs.org/projects/happypose/badge/?version=latest)](https://happypose.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/agimus-project/happypose/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/agimus-project/happypose)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Toolbox and trackers for object pose-estimation. Based on the work [CosyPose](https://github.com/Simple-Robotics/cosypose) and [MegaPose](https://github.com/megapose6d/megapose6d). This directory is currently under development. Please refer to the [documentation](https://agimus-project.github.io/happypose/) for more details.


# Installation

This installation procedure will be curated.

```
git clone --branch dev --recurse-submodules https://github.com/agimus-project/happypose.git
cd happypose
conda env create -f environment.yml
conda activate happypose
cd happypose/pose_estimators/cosypose
pip install .
cd ../../..
pip install -e .
```


# Create data directory

```
Create data dir /somewhere/convenient. The dataset to store are quite large.
export HAPPYPOSE_DATA_DIR=/somewhere/convenient
```

# Configuration for the evaluation

Installation of bop_toolkit :

```
conda activate happypose
cd happypose/deps/bop_toolkit_challenge/
# Remove all versions enforcing on requirements.txt
pip install -r requirements.txt -e .
```

If you plan on evaluating CosyPose and Megapose, you need to modify the paths to data directories used by the bop_toolkit.
A simple solution is to create subfolders in `HAPPYPOSE_DATA_DIR`.
This is done by modifying following lines in `deps/bop_toolkit_challenge/bop_toolkit_lib/config.py`, replace

```
######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  datasets_path = r'/path/to/bop/datasets'

# Folder with pose results to be evaluated.
results_path = r'/path/to/folder/with/results'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/path/to/eval/folder'
```

with

```
######## Basic ########

HAPPYPOSE_DATA_DIR = os.environ['HAPPYPOSE_DATA_DIR']

# Folder with the BOP datasets.
datasets_path = os.path.join(HAPPYPOSE_DATA_DIR, 'bop_datasets')

# Folder with pose results to be evaluated.
results_path = os.path.join(HAPPYPOSE_DATA_DIR, 'results')

# Folder for the calculated pose errors and performance scores.
eval_path = os.path.join(HAPPYPOSE_DATA_DIR, 'bop_eval_outputs')
```

You will also need to install [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) if you want to use the depth for MegaPose. To do so, please run the following commands to install it :

```
# Go to HappyPose root directory
apt install -y cmake libeigen3-dev libboost-all-dev
conda activate happypose
mamba install compilers -c conda-forge
pip install open3d
mkdir /build && cd /build && git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus && mkdir build && cd build 
cmake -DTEASERPP_PYTHON_VERSION=3.9 .. && make teaserpp_python
cd python && pip install .
```