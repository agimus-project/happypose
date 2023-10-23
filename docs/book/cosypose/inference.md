# Inference

Here are provided the minimal commands you have to run in order to run the inference of CosyPose. You need to set up the environment variable `$HAPPYPOSE_DATA_DIR` as explained in the README.

 ## 1. Download pre-trained pose estimation models

```sh
 #ycbv
python -m happypose.toolbox.utils.download --cosypose_models=detector-bop-ycbv-pbr--970850
python -m happypose.toolbox.utils.download --cosypose_models=coarse-bop-ycbv-pbr--724183
python -m happypose.toolbox.utils.download --cosypose_models=refiner-bop-ycbv-pbr--604090
```

## 2. Download YCB-V Dataset

```sh
python -m happypose.toolbox.utils.download --bop_dataset=ycbv
```

## 3. Download the example

```sh
cd $HAPPYPOSE_DATA_DIR
wget https://memmo-data.laas.fr/static/examples.tar.xz
tar xf examples.tar.xz
```

## 4. Run the script

```sh
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_inference_on_example crackers --run-inference
```

## 5. Results

The results are stored in the visualization folder created in the crackers example directory.

![Inference results](./images/all_results.png)
