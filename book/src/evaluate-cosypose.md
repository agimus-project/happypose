# Evaluating CosyPose

## Install

Two installation steps are needed : [bop_toolkit](https://github.com/thodan/bop_toolkit) and [bop_renderer](https://github.com/thodan/bop_renderer/). These repository are stored in `happypose/pose_estimators/megapose/deps/`.

### 1. Bop_toolkit_challenge

```
cd /happypose/pose_estimators/megapose/deps/bop_toolkit_challenge
# You need to remove all the versions from the requirements.txt file, then :
pip install -r requirements.txt -e .
```

Then, you need to modify the following lines in `bop_toolkit_lib/config.py`, replace :


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


<details>
<summary>This part is deprecated and will be removed </summary>

Also, replace

```
# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'
```

with
```
# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = /path/to/happypose/happypose/pose_estimators/megapose/deps/bop_renderer/build
```
</details>

### 2. Bop_renderer

<details>
<summary>This part is deprecated and will be removed</summary>


This installation is tested only on Ubuntu/Debian system. Please refer to [bop_renderer](https://github.com/thodan/bop_renderer/) if needed.

```
cd /happypose/pose_estimators/megapose/deps/bop_renderer
sudo apt install libosmesa6-dev
conda install -c conda-forge mesalib
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
</details>

## Usage

This needs to be adapted (SO3_grid not used for CosyPose)

```
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_full_cosypose_eval_new detector_run_id=bop_pbr coarse_run_id=coarse-bop-ycbv-pbr--724183 refiner_run_id=refiner-bop-ycbv-pbr--604090 ds_names=["ycbv.bop19"] result_id=ycbv-debug detection_coarse_types=[["detector","S03_grid"]]
```
