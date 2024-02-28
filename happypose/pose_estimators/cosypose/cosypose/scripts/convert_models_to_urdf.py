import argparse
from pathlib import Path

from tqdm import tqdm

from happypose.pose_estimators.cosypose.cosypose.libmesh.urdf_utils import (
    obj_to_urdf,
    ply_to_obj,
)

# from happypose.pose_estimators.cosypose.cosypose.datasets.datasets_cfg import \
#     make_object_dataset
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset


def convert_bop_dataset_to_urdfs(
    obj_ds_name: str, urdf_dir: Path, texture_size=(1024, 1024)
):
    """
    For each object, generate these files:

    {path_to_urdf_dir}/{ds_name}/{obj_label}/{obj_label}.obj
    {path_to_urdf_dir}/{ds_name}/{obj_label}/{obj_label}.obj.mtl
    {path_to_urdf_dir}/{ds_name}/{obj_label}/{obj_label}.png
    {path_to_urdf_dir}/{ds_name}/{obj_label}/{obj_label}.urdf
    """
    obj_dataset = make_object_dataset(obj_ds_name)
    urdf_ds_dir = urdf_dir / obj_ds_name
    urdf_ds_dir.mkdir(exist_ok=True, parents=True)
    for n in tqdm(range(len(obj_dataset))):
        obj = obj_dataset[n]
        ply_path = obj.mesh_path
        obj_name = ply_path.with_suffix("").name
        obj_urdf_dir = urdf_ds_dir / obj_name
        obj_urdf_dir.mkdir(exist_ok=True)
        obj_path = (obj_urdf_dir / obj_name).with_suffix(".obj")
        ply_to_obj(ply_path, obj_path, texture_size=texture_size)
        obj_to_urdf(obj_path, obj_path.with_suffix(".urdf"))


def main():
    parser = argparse.ArgumentParser("3D ply object models -> pybullet URDF converter")
    parser.add_argument("--models", default="", type=str)
    args = parser.parse_args()

    from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

    urdf_dir = LOCAL_DATA_DIR / "urdfs"
    convert_bop_dataset_to_urdfs(args.models, urdf_dir)


if __name__ == "__main__":
    main()
