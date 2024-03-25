# Inference

Here are provided the minimal commands you have to run in order to run the inference of CosyPose on the barbecu-sauce example. You need to set up the environment variable `$HAPPYPOSE_DATA_DIR` as explained in the README.

 ## 1. Download pre-trained pose estimation models

```sh
#hope dataset detector
python -m happypose.toolbox.utils.download --cosypose_models \
            detector-bop-hope-pbr--15246 \
            coarse-bop-hope-pbr--225203 \
            refiner-bop-hope-pbr--955392
```
## 2. Download the example data

```sh
python -m happypose.toolbox.utils.download --examples barbecue-sauce
```

## 3. Run the script
The example contains default outputs for detection and pose prediction
```sh
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_inference_on_example barbecue-sauce --run-inference --run-detections --vis-detections --vis-poses
```

## 4. Results

The results are stored in the visualization folder created in the example directory.

![Inference results](./images/all_results.png)
