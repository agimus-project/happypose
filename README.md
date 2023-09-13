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

Installation of bop_toolkit :

```
conda activate happypose
cd happypose/pose_estimators/megapose/deps/bop_toolkit_challenge/
# Remove all versions enforcing on requirements.txt
pip install -r requirements.txt -e .
```


# Create data directory

```
Create data dir /somewhere/convenient. The dataset to store are quite large.
export MEGAPOSE_DATA_DIR=/somewhere/convenient
```

# Configuration for the evaluation

If you plan on evaluating CosyPose and Megapose, you need to modify the following lines in `bop_toolkit_lib/config.py`, replace

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

# Folder with the BOP datasets.
datasets_path = str(os.environ['BOP_DATASETS_PATH'])
results_path = str(os.environ['BOP_RESULTS_PATH'])
eval_path = str(os.environ['BOP_EVAL_PATH'])
```