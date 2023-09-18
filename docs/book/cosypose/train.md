# Train CosyPose

Disclaimer : This part of the repository is still under development.

 Training the detector part of the pose estimation part are independant.

## Training Pose Estimator

This script can be used to train both the coarse model or the refiner model.

```
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_pose_training --config ycbv-refiner-syntonly
```

## Training Detector

```
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_detector_training --config bop-ycbv-synt+real
```

All the models were trained on 32 GPUs.