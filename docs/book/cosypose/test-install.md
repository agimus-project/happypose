# Testing your installation

 ## Download pre-trained pose estimation models

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

You need to download the ycbv dataset to run this example. Please see the download section.

```
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_inference_on_example crackers --run-inference
```
