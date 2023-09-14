# Main entry points

This repository is divided into different entry points

- [Inference](./test-install.md): `run_cosypose_on_example.py` is used to run the inference pipeline on a single example image. 
- [Evaluation](./evaluate.md): ̀`run_full_cosypose_evaluation.py` is ued to first run inference on one or several datasets, and then use the results obtained to evaluate the method on these datasets. 
- [Training](./train.md): `run_detector_training.py` is used to train the detector part of Cosypose.`run_pose_training.py` can be used to train the `coarse` model or the `refiner` model.