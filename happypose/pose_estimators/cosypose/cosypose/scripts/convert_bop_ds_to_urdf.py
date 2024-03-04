import argparse
from pathlib import Path

from happypose.pose_estimators.cosypose.cosypose.libmesh.urdf_utils import (
    convert_rigid_body_dataset_to_urdfs,
)
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset


def convert_bop_dataset_to_urdfs(
    obj_ds_name: str, urdf_dir: Path, texture_size=(1024, 1024), override=True
):
    obj_dataset = make_object_dataset(obj_ds_name)
    urdf_ds_dir = urdf_dir / obj_ds_name
    convert_rigid_body_dataset_to_urdfs(
        obj_dataset, urdf_ds_dir, texture_size, override
    )


def main():
    parser = argparse.ArgumentParser("3D ply object ds_name -> pybullet URDF converter")
    parser.add_argument(
        "--ds_name",
        default="",
        type=str,
        help="Bop dataset model name: ycbv, tless.cad, etc.",
    )
    parser.add_argument(
        "--override",
        default=True,
        type=bool,
        help="If true, erases previous content of urdf/<ds_name>.",
    )
    args = parser.parse_args()

    from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR

    urdf_dir = LOCAL_DATA_DIR / "urdfs"
    convert_bop_dataset_to_urdfs(args.ds_name, urdf_dir, args.override)


if __name__ == "__main__":
    main()
