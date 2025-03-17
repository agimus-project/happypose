"""Set of unit tests for testing inference example for CosyPose."""

import numpy as np
import pandas as pd
import pinocchio as pin
import pytest
import torch

from happypose.pose_estimators.cosypose.cosypose.config import LOCAL_DATA_DIR
from happypose.pose_estimators.megapose.inference.icp_refiner import ICPRefiner
from happypose.toolbox.datasets.bop_object_datasets import (
    RigidObject,
    RigidObjectDataset,
)
from happypose.toolbox.inference.example_inference_utils import load_observation_example
from happypose.toolbox.inference.types import ObservationTensor, PoseEstimatesType
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import (
    Panda3dBatchRenderer,
)


class TestCosyPoseInference:
    """Unit tests for CosyPose inference example."""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """Run detector with coarse and refiner from CosyPose."""
        self.expected_object_label = "hope-obj_000002"
        mesh_file_name = "hope-obj_000002.ply"
        data_dir = LOCAL_DATA_DIR / "examples" / "barbecue-sauce"
        mesh_dir = data_dir / "meshes"
        mesh_path = mesh_dir / mesh_file_name

        self.coarse_run_id = "coarse-bop-hope-pbr--225203"
        self.refiner_run_id = "refiner-bop-hope-pbr--955392"

        rgb, depth, camera_data = load_observation_example(data_dir, load_depth=True)
        self.observation = ObservationTensor.from_numpy(
            rgb, depth=depth, K=camera_data.K
        )

        self.object_dataset = RigidObjectDataset(
            objects=[
                RigidObject(
                    label=self.expected_object_label,
                    mesh_path=mesh_path,
                    mesh_units="mm",
                )
            ]
        )
        mesh_db = MeshDataBase.from_object_ds(self.object_dataset)
        self.mesh_db_batched = mesh_db.batched().cpu()

    def test_icp_refiner(self):
        renderer = Panda3dBatchRenderer(
            self.object_dataset,
            n_workers=1,
            preload_cache=False,
        )

        depth_refiner = ICPRefiner(self.mesh_db_batched, renderer)

        # hardcoded pose: obtained from the output of single rgb prediction
        T_in = pin.SE3(
            pin.exp3(np.array([1.4, 1.6, -1.11])),
            np.array([0.1, 0.07, 0.45]),
        )

        poses_in = torch.from_numpy(T_in.homogeneous)[np.newaxis, :, :]
        preds_cosy = PoseEstimatesType(
            infos=pd.DataFrame(
                {"batch_im_id": [0], "label": ["hope-obj_000002"], "score": [1.0]}
            ),
            poses=poses_in,
            poses_input=poses_in,
        )

        preds, _ = depth_refiner.refine_poses(
            predictions=preds_cosy, depth=self.observation.depth, K=self.observation.K
        )

        T_est = pin.SE3(preds.poses[0].numpy())
        diff = T_est.inverse() * T_in
        # ICP result should be only slightly different to the input
        assert np.linalg.norm(pin.log6(diff).vector) < 0.1
        # make sure input and output poses are not the same pose
        assert np.linalg.norm(pin.log6(diff).vector) > 1e-2
