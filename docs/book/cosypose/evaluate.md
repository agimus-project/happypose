# Evaluating CosyPose

Please make sure you followed the steps relative to the evaluation in the main readme.

Please run the following command to evaluate on YCBV dataset

```
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_full_cosypose_eval_new detector_run_id=bop_pbr coarse_run_id=coarse-bop-ycbv-pbr--724183 refiner_run_id=refiner-bop-ycbv-pbr--604090 ds_names=["ycbv.bop19"] result_id=ycbv-debug detection_coarse_types=[["detector","S03_grid"]]
```

The other BOP datasets are supported as long as you download the correspond models.
