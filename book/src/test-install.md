# Testing your installation

## 0. Create data directory

```
Create data dir /somewhere/convenient. The dataset to store are quite large.
export MEGAPOSE_DATA_DIR=/somewhere/convenient
cd $MEGAPOSE_DATA_DIR
wget https://memmo-data.laas.fr/static/examples.tar.xz
tar xf examples.tar.xz
```

 ## 1. Download pre-trained pose estimation models

### Megapose
Download pose estimation models to $MEGAPOSE_DATA_DIR/megapose-models:

```
python -m happypose.toolbox.utils.download --megapose_models
```

### Cosypose

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
 ## 2. Run examples

### Megapose

```
python -m happypose.pose_estimators.megapose.src.megapose.scripts.run_inference_on_example barbecue-sauce --run-inference --vis-outputs
```

### CosyPose

You need to download the ycbv dataset to run this example. Please see the download section.

```
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_inference_on_example crackers --run-inference
```
