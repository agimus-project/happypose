# HappyPose
Toolbox and trackers for object pose-estimation. Based on the work CosyPose and MegaPose. This directory is currently under development.


# Installation

This installation procedure will be curated. 

```
git clone --recurse-submodules git@github.com:agimus-project/happypose.git
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
