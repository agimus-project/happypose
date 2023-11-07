# Inference

Here are provided the minimal commands you have to run in order to run the inference of CosyPose. You need to set up the environment variable `$HAPPYPOSE_DATA_DIR` as explained in the README.

 ## 1. Download pre-trained pose estimation models

```sh
python -m happypose.toolbox.utils.download --megapose_models
```

## 2. Download the example

We estimate the pose for a barbecue sauce bottle (from the [HOPE](https://github.com/swtyree/hope-dataset) dataset, not used during training of MegaPose).

```sh
cd $HAPPYPOSE_DATA_DIR
wget https://memmo-data.laas.fr/static/examples.tar.xz
tar xf examples.tar.xz
```

The input files are the following:
```sh
$HAPPYPOSE_DATA_DIR/examples/barbecue-sauce/
    image_rgb.png
    image_depth.png
    camera_data.json
    inputs/object_data.json
    meshes/barbecue-sauce/hope_000002.ply
    meshes/barbecue-sauce/hope_000002.png
```
- `image_rgb.png` is a RGB image of the scene. We recommend using a 4:3 aspect ratio.
- `image_depth.png` (optional) contains depth measurements, with values in `mm`. You can leave out this file if you don't have depth measurements.
- `camera_data.json` contains the 3x3 camera intrinsic matrix `K` and the camera `resolution` in `[h,w]` format.

    `{"K": [[605.9547119140625, 0.0, 319.029052734375], [0.0, 605.006591796875, 249.67617797851562], [0.0, 0.0, 1.0]], "resolution": [480, 640]}`

- `inputs/object_data.json` contains a list of object detections. For each detection, the 2D bounding box in the image  (in `[xmin, ymin, xmax, ymax]` format), and the label of the object are provided. In this example, there is a single object detection. The bounding box is only used for computing an initial depth estimate of the object which is then refined by our approach. The bounding box does not need to be extremly precise (see below).

    `[{"label": "barbecue-sauce", "bbox_modal": [384, 234, 522, 455]}]`

- `meshes/barbecue-sauce` is a directory containing the object's mesh. Mesh units are expected to be in millimeters. In this example, we use a mesh in `.ply` format. The code also supports `.obj` meshes but you will have to make sure that the objects are rendered correctly with our renderer.


You can visualize input detections using :
```sh
python -m happypose.pose_estimators.megapose.scripts.run_inference_on_example barbecue-sauce --vis-detections
```

<img src="images/detections.png" width="500">


## 3. Run pose estimation and visualize results
Run inference with the following command:
```sh
python -m happypose.pose_estimators.megapose.scripts.run_inference_on_example barbecue-sauce --run-inference
```
by default, the model only uses the RGB input. You can use of our RGB-D megapose models using the `--model` argument. Please see our [Model Zoo](#model-zoo) for all models available.

The previous command will generate the following file:

```sh
$HAPPYPOSE_DATA_DIR/examples/barbecue-sauce/
    outputs/object_data.json
```

This file contains a list of objects with their estimated poses . For each object, the estimated pose is noted `TWO` (the world coordinate frame correspond to the camera frame). It is composed of a quaternion and the 3D translation:

    [{"label": "barbecue-sauce", "TWO": [[0.5453961536730983, 0.6226545207599095, -0.43295293693197473, 0.35692612413663855], [0.10723329335451126, 0.07313819974660873, 0.45735278725624084]]}]

Finally, you can visualize the results using:

```sh
python -m happypose.pose_estimators.megapose.scripts.run_inference_on_example barbecue-sauce --run-inference --vis-outputs
```
which write several visualization files:

```sh
$HAPPYPOSE_DATA_DIR/examples/barbecue-sauce/
    visualizations/contour_overlay.png
    visualizations/mesh_overlay.png
    visualizations/all_results.png
```

<img src="images/all_results.png" width="1000">
