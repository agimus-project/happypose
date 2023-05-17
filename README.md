# HappyPose

[![Tests](https://github.com/agimus-project/happypose/actions/workflows/test.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/test.yml)
[![Packaging](https://github.com/agimus-project/happypose/actions/workflows/packaging.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/packaging.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/agimus-project/happypose/main.svg)](https://results.pre-commit.ci/latest/github/agimus-project/happypose/main)
[![Documentation Status](https://readthedocs.org/projects/happypose/badge/?version=latest)](https://happypose.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/agimus-project/happypose/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/agimus-project/happypose)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Toolbox and trackers for object pose-estimation. Based on the work [CosyPose](https://github.com/Simple-Robotics/cosypose) and [MegaPose](https://github.com/megapose6d/megapose6d). This directory is currently under development.


# Installation

This installation procedure will be curated.

```
git clone --recurse-submodules https://github.com/agimus-project/happypose.git
conda env create -f environment.yml
cd happypose/happypose/pose_estimators/cosypose
python setup.py install
pip install -e .
```
# Data


```
Create data dir in happypose/pose_estimators/cosypose/local_data
export MEGAPOSE_DATA_DIR=happypose/pose_estimators/cosypose/local_data
```

download [barbecue sauce](https://drive.google.com/drive/folders/10BIvhnrKGbNr8EKGB3KUtkSNcp460k9S) and put it in `$MEGAPOSE_DATA_DIR/examples/barbecue-sauce/`

# Testing the install

```
python -m happypose.pose_estimators.megapose.src.megapose.scripts.run_inference_on_example barbecue-sauce --vis-outputs
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_inference_on_example cheetos --run-inference
```