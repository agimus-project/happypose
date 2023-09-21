
# Standard Library
from pathlib import Path
from typing import Any, Dict, Optional

# Third Party
import torch
from omegaconf import OmegaConf

# MegaPose
import happypose
import happypose.pose_estimators.megapose as megapose
import happypose.pose_estimators.megapose.evaluation.evaluation_runner
import happypose.toolbox.datasets.datasets_cfg
import happypose.toolbox.inference.utils
from happypose.pose_estimators.megapose.evaluation.eval_config import (
    EvalConfig,
)
from happypose.pose_estimators.megapose.evaluation.evaluation_runner import (
    EvaluationRunner,
)
from happypose.pose_estimators.megapose.evaluation.meters.modelnet_meters import (
    ModelNetErrorMeter,
)
from happypose.pose_estimators.cosypose.cosypose.evaluation.prediction_runner import (
    PredictionRunner,
)
from happypose.pose_estimators.megapose.evaluation.runner_utils import (
    format_results,
)
from happypose.pose_estimators.megapose.inference.depth_refiner import (
    DepthRefiner,
)
from happypose.pose_estimators.megapose.inference.icp_refiner import (
    ICPRefiner,
)
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset

# from happypose.pose_estimators.megapose.inference.teaserpp_refiner import TeaserppRefiner
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.utils.distributed import get_rank, get_tmp_dir
from happypose.toolbox.utils.logging import get_logger

# """" Temporary imports

from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR, RESULTS_DIR

# Pose estimator
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import check_update_config as check_update_config_pose
from happypose.pose_estimators.cosypose.cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_predictor import CoarseRefinePosePredictor
from happypose.pose_estimators.cosypose.cosypose.integrated.multiview_predictor import MultiviewScenePredictor
from happypose.pose_estimators.cosypose.cosypose.datasets.wrappers.multiview_wrapper import MultiViewWrapper
import cosypose.utils.tensor_collection as tc
# Detection
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import create_model_detector
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import check_update_config as check_update_config_detector
from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector

from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer

import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = get_logger(__name__)

def load_detector(run_id, ds_name):
    run_dir = EXP_DIR / run_id
    # cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / 'checkpoint.pth.tar', map_location=device)
    ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model, ds_name)
    return model

def load_pose_models(coarse_run_id, refiner_run_id, n_workers):
    run_dir = EXP_DIR / coarse_run_id
    # cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
    cfg = check_update_config_pose(cfg)
    # object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
    #object_ds = make_object_dataset(cfg.object_ds_name)
    #mesh_db = MeshDataBase.from_object_ds(object_ds)
    #renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers, gpu_renderer=gpu_renderer)
    #
    
    object_dataset = make_object_dataset("ycbv")
    mesh_db = MeshDataBase.from_object_ds(object_dataset)
    renderer = Panda3dBatchRenderer(object_dataset, n_workers=n_workers, preload_cache=False)
    mesh_db_batched = mesh_db.batched().to(device)

    def load_model(run_id):
        run_dir = EXP_DIR / run_id
        # cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
        cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.UnsafeLoader)
        cfg = check_update_config_pose(cfg)
        if cfg.train_refiner:
            model = create_model_refiner(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        else:
            model = create_model_coarse(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / 'checkpoint.pth.tar', map_location=device)
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.config = cfg
        return model
    
    coarse_model = load_model(coarse_run_id)
    refiner_model = load_model(refiner_run_id)
    return coarse_model, refiner_model, mesh_db



def generate_save_key(detection_type: str, coarse_estimation_type: str) -> str:
    return f"{detection_type}+{coarse_estimation_type}"


def get_save_dir(cfg: EvalConfig) -> Path:
    """Returns a save dir.

    Example

    .../ycbv.bop19/gt+SO3_grid

    You must remove the '.bop19' from the name in order for the
    bop_toolkit_lib to process it correctly.

    """
    save_key = generate_save_key(cfg.inference.detection_type, cfg.inference.coarse_estimation_type)

    assert cfg.save_dir is not None
    assert cfg.ds_name is not None
    save_dir = Path(cfg.save_dir) / cfg.ds_name / save_key
    return save_dir


def run_eval(
    cfg: EvalConfig,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run eval for a single setting on a single dataset.

    A single setting is a (detection_type, coarse_estimation_type) such
    as ('maskrcnn', 'SO3_grid').

    Saves the results to the directory below (if one is not passed in).

    cfg.save_dir / ds_name / eval_key / results.pth.tar

    Returns:
        dict: If you are rank_0 process, otherwise returns None

    """

    save_key = generate_save_key(cfg.inference.detection_type, cfg.inference.coarse_estimation_type)
    if save_dir is None:
        save_dir = get_save_dir(cfg)

    cfg.save_dir = str(save_dir)

    logger.info(f"Running eval on ds_name={cfg.ds_name} with setting={save_key}")

    # Load the dataset
    ds_kwargs = dict(load_depth=False)
    scene_ds = happypose.toolbox.datasets.datasets_cfg.make_scene_dataset(cfg.ds_name, **ds_kwargs)
    urdf_ds_name, obj_ds_name = happypose.toolbox.datasets.datasets_cfg.get_obj_ds_info(cfg.ds_name)

    # drop frames if this was specified
    if cfg.n_frames is not None:
        scene_ds.frame_index = scene_ds.frame_index[: cfg.n_frames].reset_index(drop=True)

    # Load detector model
    if cfg.inference.detection_type == "detector":
        assert cfg.detector_run_id is not None
        detector_model = load_detector(cfg.detector_run_id, cfg.ds_name)
    elif cfg.inference.detection_type == "gt":
        detector_model = None
    else:
        raise ValueError(f"Unknown detection_type={cfg.inference.detection_type}")

    # Load the coarse and mrefiner models
    # Needed to deal with the fact that str and Optional[str] are incompatible types.
    # See https://stackoverflow.com/a/53287330
    assert cfg.coarse_run_id is not None
    assert cfg.refiner_run_id is not None
    # TODO (emaitre): This fuction seems to take the wrong parameters. Trying to fix this
    """
    coarse_model, refiner_model, mesh_db = happypose.toolbox.inference.utils.load_pose_models(
        coarse_run_id=cfg.coarse_run_id,
        refiner_run_id=cfg.refiner_run_id,
        n_workers=cfg.n_rendering_workers,
        obj_ds_name=obj_ds_name,
        urdf_ds_name=urdf_ds_name,
        force_panda3d_renderer=True,
    )
    """
    object_ds = make_object_dataset(obj_ds_name)


    coarse_model, refiner_model, mesh_db = load_pose_models(
        coarse_run_id=cfg.coarse_run_id,
        refiner_run_id=cfg.refiner_run_id,
        n_workers=8,
    )


    renderer = refiner_model.renderer

    if cfg.inference.run_depth_refiner:
        if cfg.inference.depth_refiner == "icp":
            depth_refiner: Optional[DepthRefiner] = ICPRefiner(mesh_db, renderer)
        elif cfg.inference.depth_refiner == "teaserpp":
            depth_refiner = TeaserppRefiner(mesh_db, renderer)
        else:
            depth_refiner = None
    else:
        depth_refiner = None

    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=detector_model,
    )

    # Create the prediction runner and run inference
    assert cfg.batch_size == 1
    pred_runner = PredictionRunner(
        scene_ds=scene_ds,
        inference_cfg=cfg.inference,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_dataloader_workers,
    )

    # Run inference
    with torch.no_grad():
        all_preds = pred_runner.get_predictions(pose_estimator)

    logger.info(f"Done with inference on ds={cfg.ds_name}")
    logger.info(f"Predictions: {all_preds.keys()}")

    # Keep it simple for now. Only eval the final prediction
    eval_keys = set()
    eval_keys.add("refiner/final")
    eval_keys.add("depth_refiner")

    # Compute eval metrics
    # TODO (lmanuelli): Fix this up.
    # TODO (ylabbe): Clean this.
    eval_metrics, eval_dfs = dict(), dict()
    if not cfg.skip_evaluation:
        assert "modelnet" in cfg.ds_name
        object_ds = make_object_dataset(obj_ds_name)
        mesh_db = MeshDataBase.from_object_ds(object_ds)
        meters = {
            "modelnet": ModelNetErrorMeter(mesh_db, sample_n_points=None),
        }
        eval_runner = EvaluationRunner(
            scene_ds,
            meters,
            n_workers=cfg.n_dataloader_workers,
            cache_data=False,
            batch_size=1,
            sampler=pred_runner.sampler,
        )
        for preds_k, preds in all_preds.items():
            do_eval = preds_k in set(eval_keys)
            if do_eval:
                logger.info(f"Evaluation of predictions: {preds_k} (n={len(preds)})")
                eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            else:
                logger.info(f"Skipped: {preds_k} (n={len(all_preds)})")

    # Gather predictions from different processes
    logger.info("Waiting on barrier.")
    torch.distributed.barrier()
    logger.info("Gathering predictions from all processes.")
    for k, v in all_preds.items():
        all_preds[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()

    torch.distributed.barrier()
    logger.info("Finished gathering predictions from all processes.")

    # Save results to disk
    if get_rank() == 0:
        results_path = save_dir / "results.pth.tar"
        assert cfg.save_dir is not None
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Finished evaluation on {cfg.ds_name}, setting={save_key}")
        results = format_results(all_preds, eval_metrics, eval_dfs)
        torch.save(results, results_path)
        torch.save(results.get("summary"), save_dir / "summary.pth.tar")
        torch.save(results.get("predictions"), save_dir / "predictions.pth.tar")
        torch.save(results.get("dfs"), save_dir / "error_dfs.pth.tar")
        torch.save(results.get("metrics"), save_dir / "metrics.pth.tar")
        (save_dir / "summary.txt").write_text(results.get("summary_txt", ""))
        (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        logger.info(f"Saved predictions+metrics in {save_dir}")

        return {
            "results": results,
            "pred_keys": list(all_preds.keys()),
            "save_dir": save_dir,
            "results_path": results_path,
        }
    else:
        return None
