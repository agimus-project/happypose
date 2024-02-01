import numpy as np
import torch
from tqdm import tqdm

from happypose.pose_estimators.cosypose.cosypose.datasets.datasets_cfg import (
    make_urdf_dataset,
)
from happypose.toolbox.renderer.bullet_scene_renderer import BulletSceneRenderer

if __name__ == "__main__":
    # ds_name = 'hb'
    ds_name = "ycbv"
    urdf_ds = make_urdf_dataset(ds_name)
    renderer = BulletSceneRenderer(urdf_ds, gpu_renderer=False)
    TCO = torch.tensor(
        [
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [-1, 0, 0, 0.3],
            [0, 0, 0, 1],
        ],
    ).numpy()

    fx, fy = 300, 300
    cx, cy = 320, 240
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
    )
    cam = {
        "resolution": (640, 480),
        "K": K,
        "TWC": np.eye(4),
    }

    all_images = []
    labels = renderer.urdf_ds.index["label"].tolist()
    for _n, obj_label in tqdm(enumerate(np.random.permutation(labels))):
        obj = {
            "name": obj_label,
            "TWO": TCO,
        }
        renders = renderer.render_scene([obj], [cam])[0]["rgb"]
        assert renders.sum() > 0, obj_label
