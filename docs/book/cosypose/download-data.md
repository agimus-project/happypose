# Downloading and preparing the data


All data used (datasets, models, results, ...) are stored in a directory `$MEGAPOSE_DATA_DIR` that you created in the Readsme. We provide the utilities for downloading required data and models. All of the files can also be [downloaded manually](https://www.paris.inria.fr/archive_ylabbeprojectsdata/).

## BOP Datasets

For both T-LESS and YCB-Video, we use the datasets in the [BOP format](https://bop.felk.cvut.cz/datasets/). If you already have them on your disk, place them in `$MEGAPOSE_DATA_DIR/bop_datasets`. Alternatively, you can download it using :

```sh
python -m happypose.toolbox.utils.download --bop_dataset=ycbv
python -m happypose.toolbox.utils.download --bop_dataset=tless
```

Additional files that contain information about the datasets used to fairly compare with prior works on both datasets.

```sh
python -m happypose.toolbox.utils.download --bop_extra_files=ycbv
python -m happypose.toolbox.utils.download --bop_extra_files=tless
```

We use [pybullet](https://pybullet.org/wordpress/) for rendering images which requires object models to be provided in the URDF format. We provide converted URDF files, they can be downloaded using:

```sh
python -m happypose.toolbox.utils.download --urdf_models=ycbv
python -m happypose.toolbox.utils.download --urdf_models=tless.cad
```

In the BOP format, the YCB objects `002_master_chef_can` and `040_large_marker` are considered symmetric, but not by previous works such as PoseCNN, PVNet and DeepIM. To ensure a fair comparison (using ADD instead of ADD-S for ADD-(S) for these objects), these objects must *not* be considered symmetric in the evaluation. To keep the uniformity of the models format, we generate a set of YCB objects `models_bop-compat_eval` that can be used to fairly compare our approach against previous works. You can download them directly:

```sh
python -m happypose.toolbox.utils.download --ycbv_compat_models
```

Notes:

- The URDF files were obtained using these commands (requires `meshlab` to be installed):

  ```sh
  python -m happypose.pose_estimators.cosypose.cosypose.scripts.convert_models_to_urdf --models=ycbv
  python -m happypose.pose_estimators.cosypose.cosypose.scripts.convert_models_to_urdf --models=tless.cad
  ```

- Compatibility models were obtained using the following script:

  ```sh
  python -m happypose.pose_estimators.cosypose.cosypose.scripts.make_ycbv_compat_models
  ```

## Models for minimal version

```sh
 #ycbv
  python -m happypose.toolbox.utils.download --cosypose_model=detector-bop-ycbv-pbr--970850
  python -m happypose.toolbox.utils.download --cosypose_model=coarse-bop-ycbv-pbr--724183
  python -m happypose.toolbox.utils.download --cosypose_model=refiner-bop-ycbv-pbr--604090

 #tless
  python -m happypose.toolbox.utils.download --cosypose_model=detector-bop-tless-pbr--873074
  python -m happypose.toolbox.utils.download --cosypose_model=coarse-bop-tless-pbr--506801
  python -m happypose.toolbox.utils.download --cosypose_model=refiner-bop-tless-pbr--233420
```

## Pre-trained models for single-view estimator

The pre-trained models of the single-view pose estimator can be downloaded using:


```sh
# YCB-V Single-view refiner
python -m happypose.toolbox.utils.download --cosypose_model=ycbv-refiner-finetune--251020

# YCB-V Single-view refiner trained on synthetic data only
# Only download this if you are interested in retraining the above model
python -m happypose.toolbox.utils.download --cosypose_model=ycbv-refiner-syntonly--596719

# T-LESS coarse and refiner models
python -m happypose.toolbox.utils.download --cosypose_model=tless-coarse--10219
python -m happypose.toolbox.utils.download --cosypose_model=tless-refiner--585928
```

## 2D detections

To ensure a fair comparison with prior works on both datasets, we use the same detections as DeepIM (from PoseCNN) on YCB-Video and the same as Pix2pose (from a RetinaNet model) on T-LESS. Download the saved 2D detections for both datasets using

```sh
python -m happypose.toolbox.utils.download --detections=ycbv_posecnn

# SiSo detections: 1 detection with highest per score per class per image on all images
# Available for each image of the T-LESS dataset (primesense sensor)
# These are the same detections as used in Pix2pose's experiments
python -m happypose.toolbox.utils.download --detections=tless_pix2pose_retinanet_siso_top1

# ViVo detections: All detections for a subset of 1000 images of T-LESS.
# Used in our multi-view experiments.
python -m happypose.toolbox.utils.download --detections=tless_pix2pose_retinanet_vivo_all
```

If you are interested in re-training a detector, please see the BOP 2020 section.

Notes:

- The PoseCNN detections (and coarse pose estimates) on YCB-Video were extracted and converted from [these PoseCNN results](https://github.com/yuxng/YCB_Video_toolbox/blob/master/results_PoseCNN_RSS2018.zip).
- The Pix2pose detections were extracted using [pix2pose's](https://github.com/kirumang/Pix2Pose) code. We used the detection model from their paper, see [here](https://github.com/kirumang/Pix2Pose#download-pre-trained-weights). For the ViVo detections, their code was slightly modified. The code used to extract detections can be found [here](https://github.com/ylabbe/pix2pose_cosypose).

</details>