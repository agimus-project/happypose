# Requirements

- Install `bop_toolkit_lib` with the `dataset-tools` branch:
```
pip install git+https://github.com/ylabbe/bop_renderer@dataset-tools
```

- Install the bop renderer for generating masks and ground truth infos

```
export WORK_DIR=/path/to/dir
cd $WORK_DIR
git clone https://github.com/ylabbe/bop_renderer && \
  export OSMESA_PREFIX=$WORK_DIR/osmesa && \
  export LLVM_PREFIX=$WORK_DIR/llvm && \
  mkdir -p $OSMESA_PREFIX && mkdir -p $LLVM_PREFIX && \
  conda install -c conda-forge autoconf automake libtool pkg-config cmake zlib -y && \
  export PYTHON_PREFIX=$CONDA_PREFIX && \
  cd $WORK_DIR/bop_renderer && mkdir -p osmesa-install/build && \
  cd osmesa-install/build && bash ../osmesa-install.sh && \
  cp -r $LLVM_PREFIX/lib/* $CONDA_PREFIX/lib && \
  cp -r $OSMESA_PREFIX/lib/* $CONDA_PREFIX/lib && \
  cd $WORK_DIR/bop_renderer && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && \
  cp *.so $CONDA_PREFIX/lib/python3.9/site-packages
```

- Install Blender, the datasets were generated with blender 2.39.8:

```
cd $HOME && \
  wget https://mirrors.dotsrc.org/blender/release/Blender2.93/blender-2.93.8-linux-x64.tar.xz && \
  tar -xvf blender-2.93.8-linux-x64.tar.xz && rm blender-2.93.8-linux-x64.tar.xz
```

- Download the `cctextures` dataset to `MEGAPOSE_DATA_DIR`.

- Install BlenderProc, currently it has only been tested with this fork

```
git clone https://github.com/ylabbe/blenderproc
```

# Testing the install
```
export BLENDER_PROC_DIR=/path/to/blenderproc
export MEGAPOSE_DATA_DIR=/path/to/megapose_data_dir
export BLENDER_INSTALL_DIR=/path/to/blender-2.93.8-linux-x64
python -m happypose.pose_estimators.megapose.src.megapose.scripts.generate_shapenet_pbr dataset_id=shapenet_1M verbose=True few=True debug=True overwrite=True
```


# Known issues
- If you try to record a dataset on the nodes of a cluster that doesn't have an internet connexion, BlenderProc may crash because it cannot download some dependencies. To solve this, you will need to run the testing script on a node that has an internet connexion, then comment out the content of the method `setup_pip` of the `SetupUtilty.py` found in blenderproc.
