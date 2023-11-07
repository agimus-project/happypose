# Main entry points

This repository is divided into different entry points

- [Inference](./test-install.md): `run_megapose_on_example.py` is used to run the inference pipeline on a single example image.
- [Evaluation](./evaluate.md): `run_full_megapose_eval.py` is ued to first run inference on one or several datasets, and then use the results obtained to evaluate the method on these datasets.

# Model Zoo

| Model name                            | Input |
|---------------------------------------|-------|
| megapose-1.0-RGB                      | RGB   |
| megapose-1.0-RGBD                     | RGB-D |
| megapose-1.0-RGB-multi-hypothesis     | RGB   |
| megapose-1.0-RGB-multi-hypothesis-icp | RGB-D |

- `megapose-1.0-RGB` and `megapose-1.0-RGBD` correspond to method presented and evaluated in the paper.
- `-multi-hypothesis` is a variant of our approach which:
    - Uses the coarse model, extracts top-K hypotheses (by default K=5);
    - For each hypothesis runs K refiner iterations;
    - Evaluates refined hypotheses using score from coarse model and selects the highest scoring one.
- `-icp` indicates running ICP refinement on the depth data.

For optimal performance, we recommend using `megapose-1.0-RGB-multi-hypothesis` for an RGB image and `megapose-1.0-RGB-multi-hypothesis-icp` for an RGB-D image. An extended paper with full evaluation of these new approaches is coming soon.
