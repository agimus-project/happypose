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
conda create env -n happypose
cd happypose/cosypose
conda env update --name happypose --file environment_noversion.yml
git lfs pull
python setup.py install

cd ../megapose6d
conda env update --name happypose --file environment_full.yml
pip install -e .
```
One might have to run again `python setup.py install` in the `cosypose` folder after the installation of megapose. After the installation, make sur to create `local_data` in both the cosypose and megapose directories.

# Usage

In the current state of this repository `megapose` should be usable as it. Concerning `cosypose`, please use the `minimal_version*.py` files. Other files may not be supported yet.
