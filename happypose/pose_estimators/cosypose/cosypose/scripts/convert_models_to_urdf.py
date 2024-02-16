import argparse
from pathlib import Path

from tqdm import tqdm

from happypose.pose_estimators.cosypose.cosypose.datasets.datasets_cfg import (
    make_object_dataset,
)
from happypose.pose_estimators.cosypose.cosypose.libmesh.urdf_utils import (
    obj_to_urdf,
    ply_to_obj,
)


def convert_bop_dataset_to_urdfs(
    obj_ds_name: str, urdf_dir: Path, texture_size=(1024, 1024)
):
    obj_dataset = make_object_dataset(obj_ds_name)
    urdf_dir.mkdir(exist_ok=True, parents=True)
    for n in tqdm(range(len(obj_dataset))):
        obj = obj_dataset[n]
        ply_path = Path(obj["mesh_path"])
        out_dir = urdf_dir / obj["label"]
        out_dir.mkdir(exist_ok=True)
        obj_path = out_dir / ply_path.with_suffix(".obj").name
        ply_to_obj(ply_path, obj_path, texture_size=texture_size)
        obj_to_urdf(obj_path, obj_path.with_suffix(".urdf"))


def main():
    parser = argparse.ArgumentParser("3D ply object models -> pybullet URDF converter")
    parser.add_argument("--models", default="", type=str)
    args = parser.parse_args()

    from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

    urdf_dir = LOCAL_DATA_DIR / "urdf"
    convert_bop_dataset_to_urdfs(urdf_dir, args.models)


if __name__ == "__main__":
    main()
