 # Download example data for minimal testing

```sh
cd $HAPPYPOSE_DATA_DIR
wget https://memmo-data.laas.fr/static/examples.tar.xz
tar xf examples.tar.xz
```

 # Download pre-trained pose estimation models

Download pose estimation models to `$HAPPYPOSE_DATA_DIR/megapose-models`:

```sh
python -m happypose.toolbox.utils.download --megapose_models
```

# Download pre-trained detection models
Megapose can use pretrained detectors from CosyPose, which can be downloaded to `$HAPPYPOSE_DATA_DIR/experiments`:

```sh
# hope
python -m happypose.toolbox.utils.download --cosypose_models \
          detector-bop-hope-pbr--15246 \
          coarse-bop-hope-pbr--225203 \
          refiner-bop-hope-pbr--955392

# ycbv

python -m happypose.toolbox.utils.download --cosypose_models \
          detector-bop-ycbv-pbr--970850 \
          coarse-bop-ycbv-pbr--724183 \
          refiner-bop-ycbv-pbr--604090

# tless
python -m happypose.toolbox.utils.download --cosypose_models \
          detector-bop-tless-pbr--873074 \
          coarse-bop-tless-pbr--506801 \
          refiner-bop-tless-pbr--233420
```

# Dataset

## Dataset information
The dataset is available at this [url](https://drive.google.com/drive/folders/1CXc_GG11jNVMeGr-Mb4o4iiNjYeKDkKd?usp=sharing). It is split into two datasets: `gso_1M` (Google Scanned Objects) and `shapenet_1M` (ShapeNet objects). Each dataset has 1 million images which were generated using [BlenderProc](https://github.com/DLR-RM/BlenderProc).

Datasets are released in the [webdataset](https://github.com/webdataset/webdataset) format for high reading performance. Each dataset is split into chunks of size ~600MB containing 1000 images each.

We provide the pre-processed meshes ready to be used for rendering and training in this [directory](https://drive.google.com/drive/folders/1AYxkv7jpDniOnTcMAxiWbdhPo8WBJaZG):
- `google_scanned_objects.zip`
- `shapenetcorev2.zip`

**Important**: Before downloading this data, please make sure you are allowed to use these datasets i.e. you can download the original ones.

## Usage
We provide utilies for loading and visualizing the data.

The following commands download 10 chunks of each dataset as well as metadatas:

```
cd $HAPPYPOSE_DATA_DIR
rclone copyto megapose_public_readonly:/webdatasets/ webdatasets/ --include "0000000*.tar" --include "*.json" --include "*.feather" --config $HAPPYPOSE_DATA_DIR/rclone.conf -P
```

We then download the object models (please make sure you have access to the original datasets before downloading these preprocessed ones):

```
cd $HAPPYPOSE_DATA_DIR
rclone copyto megapose_public_readonly:/tars tars/ --include "shapenetcorev2.zip" --include "google_scanned_objects.zip" --config $HAPPYPOSE_DATA_DIR/rclone.conf -P
unzip tars/shapenetcorev2.zip
unzip tars/google_scanned_objects.zip
```

Your directory structure should look like this:
```
$HAPPYPOSE_DATA_DIR/
    webdatasets/
        gso_1M/
            infos.json
            frame_index.feather
            00000001.tar
            ...
        shapenet_1M/
            infos.json
            frame_index.feather
            00000001.tar
            ...
    shapenetcorev2/
        ...
    googlescannedobjects/
        ...
```

You can then use the [`render_megapose_dataset.ipynb`](notebooks/render_megapose_dataset.ipynb) notebook to load and visualize the data and 6D pose annotations.

<img src="images/dataset_renders.png" width="1200">

<img src="images/dataset_renders_2.png" width="1200">
