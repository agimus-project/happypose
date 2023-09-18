# Inference

Here are provided the minimal commands you have to run in order to run the inference of CosyPose. You need to set up the environment variable `$HAPPYPOSE_DATA_DIR` as explained in the README. 

 ## 1. Download pre-trained pose estimation models

```sh
python -m happypose.toolbox.utils.download --megapose_models
```

## 2. Download the example

```sh
cd $HAPPYPOSE_DATA_DIR
wget https://memmo-data.laas.fr/static/examples.tar.xz
tar xf examples.tar.xz 
```


 ## 3. Run the script

```sh
python -m happypose.pose_estimators.megapose.src.megapose.scripts.run_inference_on_example barbecue-sauce --run-inference --vis-outputs
```
