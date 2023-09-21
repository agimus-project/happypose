# Requirements

- Install `bop_toolkit_lib` with the `dataset-tools` branch:
```
pip install git+https://github.com/ylabbe/bop_renderer@dataset-tools
```

- Install the bop renderer for generating masks and ground truth infos

```
export CC=x86_64-conda_cos6-linux-gnu-gcc &&\
export GCC=x86_64-conda_cos6-linux-gnu-g++ &&\
export GXX=x86_64-conda_cos6-linux-gnu-g++ &&\
git clone https://github.com/ylabbe/bop_renderer && \
  export OSMESA_PREFIX=$WORK_DIR/osmesa && \
  export LLVM_PREFIX=$WORK_DIR/llvm && \
  mkdir -p $OSMESA_PREFIX && mkdir -p $LLVM_PREFIX && \
  export PYTHON_PREFIX=$CONDA_PREFIX && \
  cd $WORK_DIR/bop_renderer && mkdir -p osmesa-install/build && \
  cd osmesa-install/build && bash ../osmesa-install.sh && \
  cp -r $LLVM_PREFIX/lib/* $CONDA_PREFIX/lib && \
  cp -r $OSMESA_PREFIX/lib/* $CONDA_PREFIX/lib && \
  cd $WORK_DIR/bop_renderer && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && \
  cp *.so $CONDA_PREFIX/lib/python3.9/site-packages
```
If you are using gcc>10, you way need to apply [this patch](https://cgit.freedesktop.org/mesa/mesa/diff/?id=8dacf5f9d1df95c768016a1b92465bbabed37b54).

- Install Blender, the datasets were generated with blender 2.39.8:

```
cd $HOME && \
  wget https://mirrors.dotsrc.org/blender/release/Blender2.93/blender-2.93.8-linux-x64.tar.xz && \
  tar -xvf blender-2.93.8-linux-x64.tar.xz && rm blender-2.93.8-linux-x64.tar.xz
```

- Download the `cctextures` dataset to `HAPPYPOSE_DATA_DIR`.

- Install BlenderProc, currently it has only been tested with this fork

```
git clone https://github.com/ylabbe/blenderproc
```

# Testing the install
```
export HP_DATA_DIR=/path/to/hp_data_dir
python generate_dataset.py "run_dsgen=[gso_1M,fastrun]" "job_env@runner.job_env=[happypose,lda]"
```
Please check the configuration is correct using `-c job`


# Running dataset recording on Jean-Zay
````
python generate_dataset.py "run_dsgen=[gso_1M]" "job_env@runner.job_env=[happypose,lda]" runner.use_slurm=true
```

# Post-processing script
See `experiments/postprocess_dataset.py`:
- Convert object ids from strings to integers to match the BOP format.
- Check that all keys have all annotations (there may be some keys which have missing annotations due to a job being killed during copy of the files when recording).

# Known issues
- If you try to record a dataset on the nodes of a cluster that doesn't have an internet connexion, BlenderProc may crash because it cannot download some dependencies. To solve this, you will need to run the testing script on a node that has an internet connexion, then comment out the content of the method `setup_pip` of the `SetupUtilty.py` found in blenderproc.
