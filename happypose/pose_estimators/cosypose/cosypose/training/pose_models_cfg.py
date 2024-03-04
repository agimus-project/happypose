from pathlib import Path

import torch
import yaml

# Backbones
from happypose.pose_estimators.cosypose.cosypose.models.efficientnet import EfficientNet
from happypose.pose_estimators.cosypose.cosypose.models.flownet import (
    flownet_pretrained,
)

# Pose models
from happypose.pose_estimators.cosypose.cosypose.models.pose import PosePredictor
from happypose.pose_estimators.cosypose.cosypose.models.wide_resnet import (
    WideResNet18,
    WideResNet34,
)
from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger
from happypose.toolbox.lib3d.rigid_mesh_database import BatchedMeshes

logger = get_logger(__name__)


def check_update_config(config):
    if not hasattr(config, "init_method"):
        config.init_method = "v0"
    return config


def create_pose_model_cosypose(cfg, renderer, mesh_db):
    n_inputs = 6
    backbone_str = cfg.backbone_str
    if backbone_str == "efficientnet-b3":
        backbone = EfficientNet.from_name("efficientnet-b3", in_channels=n_inputs)
        backbone.n_features = 1536
    elif backbone_str == "flownet":
        backbone = flownet_pretrained(n_inputs=n_inputs)
        backbone.n_features = 1024
    elif "resnet34" in backbone_str:
        backbone = WideResNet34(n_inputs=n_inputs)
    elif "resnet18" in backbone_str:
        backbone = WideResNet18(n_inputs=n_inputs)
    else:
        msg = "Unknown backbone"
        raise ValueError(msg, backbone_str)

    pose_dim = cfg.n_pose_dims

    logger.info(f"Backbone: {backbone_str}")
    backbone.n_inputs = n_inputs
    render_size = (240, 320)
    model = PosePredictor(
        backbone=backbone,
        renderer=renderer,
        mesh_db=mesh_db,
        render_size=render_size,
        pose_dim=pose_dim,
    )
    return model


def load_model_cosypose(
    run_dir: Path, renderer, mesh_db_batched: BatchedMeshes, device
):
    cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader)
    cfg = check_update_config(cfg)
    model = create_pose_model_cosypose(cfg, renderer=renderer, mesh_db=mesh_db_batched)
    ckpt = torch.load(run_dir / "checkpoint.pth.tar", map_location=device)
    ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    model.cfg = cfg
    model.config = cfg
    return model
