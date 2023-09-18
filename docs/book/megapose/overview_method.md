# MegaPose
This repository contains code, models and dataset for our MegaPose paper. 

Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, Josef Sivic. “MegaPose: 6D Pose Estimation of Novel Objects via Render & Compare.” In: CoRL 2022.

[[Paper]](https://arxiv.org/abs/2212.06870) [[Project page]](https://megapose6d.github.io/)

## News
- **09.01.2023** We release two new variants of our approach (see the [Model Zoo](#model-zoo)).
- **09.01.2023** Code, models and dataset are released in this repository.
- **10.09.2022** The paper is accepted at CoRL 2022.

## Contributors
The main contributors to the code are:
- [Yann Labbé](https://ylabbe.github.io/) (Inria, NVIDIA internship)
- [Lucas Manuelli](https://lucasmanuelli.com) (NVIDIA Seattle Robotics Lab)

## Citation
If you find this source code useful please cite:

```
@inproceedings{labbe2022megapose,
  title = {{{MegaPose}}: {{6D Pose Estimation}} of {{Novel Objects}} via {{Render}} \& {{Compare}}},
  booktitle = {CoRL},
  author = {Labb\'e, Yann and Manuelli, Lucas and Mousavian, Arsalan and Tyree, Stephen and Birchfield, Stan and Tremblay, Jonathan and Carpentier, Justin and Aubry, Mathieu and Fox, Dieter and Sivic, Josef},
  date = {2022}
}
```

# Overview
This repository contains pre-trained models for pose estimation of novel objects, and our synthetic training dataset. Most notable features are listed below.

## Pose estimation of novel objects
<img src="./images/pose-estimation.png" width="800">

We provide pre-trained models for 6D pose estimation of novel objects. 

Given as inputs: 
- an RGB image (depth can also be used but is optional),
- the intrinsic parameters of the camera,
- a mesh of the object,
- a bounding box of that object in the image,

our approach estimates the 6D pose of the object (3D rotation + 3D translation) with respect to the camera. 

We provide a script and an example for inference on novel objects. After installation, please see the [Inference tutorial](#inference-tutorial).

## Large-scale synthetic dataset
<img src="./images/dataset.jpg" width="800">

We provide the synthetic dataset we used to train MegaPose. The dataset contains 2 million images displaying more than 20,000 objects from the Google Scanned Objects and ShapeNet datasets. After installation, please see the [Dataset section](#dataset).