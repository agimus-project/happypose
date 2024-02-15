import multiprocessing
from typing import List, Union

import numpy as np
import torch

from happypose.toolbox.datasets.datasets_cfg import RigidObjectDataset, UrdfDataset
from happypose.toolbox.lib3d.transform_ops import invert_transform_matrices
from happypose.toolbox.renderer.bullet_scene_renderer import BulletSceneRenderer
from happypose.toolbox.renderer.types import BatchRenderOutput, WorkerRenderOutput


def init_renderer(
    asset_dataset: Union[RigidObjectDataset, UrdfDataset],
    preload=True,
    gpu_renderer=True,
):
    renderer = BulletSceneRenderer(
        asset_dataset=asset_dataset,
        preload_cache=preload,
        background_color=(0, 0, 0),
        gpu_renderer=gpu_renderer,
    )
    return renderer


def worker_loop(
    worker_id,
    in_queue,
    out_queue,
    asset_dataset,
    preload=True,
    gpu_renderer=True,
):
    renderer = init_renderer(asset_dataset, preload=preload, gpu_renderer=gpu_renderer)
    while True:
        render_args = in_queue.get()
        if render_args is None:
            return

        obj_infos = render_args["obj_infos"]
        cam_infos = render_args["cam_infos"]
        render_depth = render_args["render_depth"]
        render_binary_mask = render_args["render_binary_mask"]
        is_valid = (
            np.isfinite(obj_infos[0]["TWO"]).all()
            and np.isfinite(cam_infos[0]["TWC"]).all()
            and np.isfinite(cam_infos[0]["K"]).all()
        )
        if is_valid:
            renderings = renderer.render_scene(
                obj_infos=obj_infos,
                cam_infos=cam_infos,
                render_depth=render_depth,
                render_binary_mask=render_binary_mask,
            )
            rgbs = np.stack([ren.rgb for ren in renderings])
            depth = (
                np.stack([ren.depth for ren in renderings]) if render_depth else None
            )
            binary_mask = (
                np.stack([ren.binary_mask for ren in renderings])
                if render_binary_mask
                else None
            )
        else:
            w, h = cam_infos[0]["resolution"]
            rgbs = np.zeros((1, h, w, 3), dtype=np.uint8)
            depth = np.zeros((1, h, w), dtype=np.float32)
            binary_mask = np.zeros((1, h, w), dtype=bool)

        output = WorkerRenderOutput(
            data_id=render_args["data_id"],
            rgb=rgbs,
            normals=None,
            depth=depth,
            binary_mask=binary_mask,
        )
        out_queue.put(output)


class BulletBatchRenderer:
    def __init__(
        self,
        asset_dataset: Union[RigidObjectDataset, UrdfDataset],
        n_workers=8,
        preload_cache=True,
        gpu_renderer=True,
    ):
        self.asset_dataset = asset_dataset
        self.n_workers = n_workers
        self.init_plotters(preload_cache, gpu_renderer)
        self.gpu_renderer = gpu_renderer

    def render(
        self,
        labels: List[str],
        TCO: torch.Tensor,
        K: torch.Tensor,
        resolution=(240, 320),
        render_depth: bool = False,
        render_binary_mask: bool = False,
    ):
        TCO = torch.as_tensor(TCO).detach()
        TOC = invert_transform_matrices(TCO).cpu().numpy()
        K = torch.as_tensor(K).cpu().numpy()

        bsz = len(labels)
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)

        # ==================================
        # Send batches of renders to workers
        # ==================================
        # NOTE: Could be faster with pytorch 3.8's sharedmemory
        for n in np.arange(bsz):
            obj_info = {
                "name": labels[n],
                "TWO": np.eye(4),
            }
            cam_info = {
                "resolution": resolution,
                "K": K[n],
                "TWC": TOC[n],
            }
            render_args = {
                "cam_infos": [cam_info],
                "obj_infos": [obj_info],
                "render_depth": render_depth,
                "render_binary_mask": render_binary_mask,
            }
            if self.n_workers > 0:
                render_args["data_id"] = n
                self.in_queue.put(render_args)
            else:
                renderings = self.plotters[0].render_scene(**render_args)
                # by definition, each "scene" in batch rendering corresponds to 1 camera, 1 object
                # -> retrieves the first and only rendering
                renderings_ = renderings[0]

                output = WorkerRenderOutput(
                    data_id=n,
                    rgb=renderings_.rgb,
                    normals=None,
                    depth=renderings_.depth if render_depth else None,
                    binary_mask=renderings_.binary_mask if render_binary_mask else None,
                )

                self._out_queue.put(output)

        # ===============================
        # Retrieve the workers renderings
        # ===============================
        list_rgbs = [None for _ in np.arange(bsz)]
        list_depths = [None for _ in np.arange(bsz)]
        list_binary_masks = [None for _ in np.arange(bsz)]

        for n in np.arange(bsz):
            renders: WorkerRenderOutput = self._out_queue.get()
            data_id = renders.data_id
            list_rgbs[data_id] = torch.tensor(renders.rgb)
            if render_depth:
                list_depths[data_id] = torch.tensor(renders.depth)
            if render_binary_mask:
                list_binary_masks[data_id] = torch.tensor(renders.binary_mask)
            del renders

        assert list_rgbs[0] is not None

        if self.gpu_renderer:
            rgbs = torch.stack(list_rgbs).pin_memory().cuda(non_blocking=True)
        else:
            rgbs = torch.stack(list_rgbs)
        rgbs = rgbs.float().permute(0, 3, 1, 2) / 255

        depths = None
        binary_masks = None

        if render_depth:
            assert list_depths[0] is not None
            if torch.cuda.is_available():
                depths = torch.stack(list_depths).pin_memory().cuda(non_blocking=True)
            else:
                depths = torch.stack(list_depths)
            depths = depths.float().permute(0, 3, 1, 2)

        if render_binary_mask:
            assert list_binary_masks[0] is not None
            if torch.cuda.is_available():
                binary_masks = (
                    torch.stack(list_binary_masks).pin_memory().cuda(non_blocking=True)
                )
            else:
                binary_masks = torch.stack(list_binary_masks)
            binary_masks = binary_masks.permute(0, 3, 1, 2)

        return BatchRenderOutput(
            rgbs=rgbs,
            normals=None,
            depths=depths,
            binary_masks=binary_masks,
        )

    def init_plotters(self, preload_cache: bool, gpu_renderer: bool):
        self.plotters = []
        self.in_queue = multiprocessing.Queue()
        self._out_queue = multiprocessing.Queue()

        if self.n_workers > 0:
            for n in range(self.n_workers):
                plotter = multiprocessing.Process(
                    target=worker_loop,
                    render_args={
                        "worker_id": n,
                        "in_queue": self.in_queue,
                        "out_queue": self._out_queue,
                        "asset_dataset": self.asset_dataset,
                        "preload": preload_cache,
                        "gpu_renderer": gpu_renderer,
                    },
                )
                plotter.start()
                self.plotters.append(plotter)
        else:
            self.plotters = [
                init_renderer(self.asset_dataset, preload_cache, gpu_renderer),
            ]

    def stop(self):
        if self.n_workers > 0:
            for _p in self.plotters:
                self.in_queue.put(None)
            for p in self.plotters:
                p.join()
                p.terminate()
        self.in_queue.close()
        self._out_queue.close()

    def __del__(self):
        self.stop()
