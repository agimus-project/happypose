# Evaluating Megapose

Please make sure you followed the steps relative to the evaluation in the main readme.

An example to run the evaluation on `YCBV` dataset. Several datasets can be added to the list.

```
python -m happypose.pose_estimators.megapose.src.megapose.scripts.run_full_megapose_eval detector_run_id=bop_pbr coarse_run_id=coarse-rgb-906902141 refiner_run_id=refiner-rgb-653307694 ds_names=[ycbv.bop19] result_id=fastsam_kbestdet_1posehyp detection_coarse_types=[["sam","SO3_grid"]] inference.n_pose_hypotheses=1 skip_inference=true run_bop_eval=true
```
