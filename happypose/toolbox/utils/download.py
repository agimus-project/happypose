#!/usr/bin/env python
import argparse
import asyncio
import logging
import os
import random
import re
import zipfile
from pathlib import Path
from typing import Set

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

from happypose.pose_estimators.cosypose.cosypose.config import (
    BOP_DS_DIR,
    LOCAL_DATA_DIR,
)
from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger

logger = get_logger(__name__)

MIRRORS = {
    "inria": "https://www.paris.inria.fr/archive_ylabbeprojectsdata/",
    "laas": "https://gepettoweb.laas.fr/data/happypose/",
    "bop": "https://bop.felk.cvut.cz/media/data/bop_datasets/",
}

DOWNLOAD_DIR = LOCAL_DATA_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)
BOP_DATASETS = {
    "ycbv": {
        "splits": ["train_real", "train_synt", "test_all"],
    },
    "tless": {
        "splits": ["test_primesense_all", "train_primesense"],
    },
    "hb": {
        "splits": ["test_primesense_all", "val_primesense"],
    },
    "icbin": {
        "splits": ["test_all"],
    },
    "itodd": {
        "splits": ["val", "test_all"],
    },
    "lm": {
        "splits": ["test_all"],
    },
    "lmo": {
        "splits": ["test_all"],
        "has_pbr": False,
    },
    "tudl": {
        "splits": ["test_all", "train_real"],
    },
    "hope": {
        "splits": ["test_all"],
    },
}

BOP_DS_NAMES = list(BOP_DATASETS.keys())


async def main():
    parser = argparse.ArgumentParser("HappyPose download utility")
    parser.add_argument("--bop_dataset", nargs="*", choices=BOP_DS_NAMES)
    parser.add_argument("--bop_extra_files", nargs="*", choices=["ycbv", "tless"])
    parser.add_argument("--cosypose_models", nargs="*")
    parser.add_argument("--megapose_models", action="store_true")
    parser.add_argument("--urdf_models", nargs="*")
    parser.add_argument("--ycbv_compat_models", action="store_true")
    parser.add_argument("--ycbv_tests", action="store_true")
    parser.add_argument("--texture_dataset", action="store_true")
    parser.add_argument("--result_id", nargs="*")
    parser.add_argument("--bop_result_id", nargs="*")
    parser.add_argument("--synt_dataset", nargs="*")
    parser.add_argument("--detections", nargs="*")
    parser.add_argument("--examples", nargs="*")
    parser.add_argument("--mirror", default="")
    parser.add_argument("--example_scenario", action="store_true")
    parser.add_argument("--pbr_training_images", action="store_true")
    parser.add_argument("--all_bop20_results", action="store_true")
    parser.add_argument("--all_bop20_models", action="store_true")

    to_dl = []
    to_symlink = []
    to_unzip = []

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.bop_dataset:
        for dataset in args.bop_dataset:
            to_dl.append((f"{dataset}_base.zip", BOP_DS_DIR / dataset))
            download_pbr = args.pbr_training_images and BOP_DATASETS[dataset].get(
                "has_pbr", True
            )
            suffixes = ["models"] + BOP_DATASETS[dataset]["splits"]
            if download_pbr:
                suffixes += ["train_pbr"]
            for suffix in suffixes:
                to_dl.append(
                    (
                        f"{dataset}_{suffix}.zip",
                        BOP_DS_DIR / dataset,
                    )
                )

    if args.bop_extra_files:
        for extra in args.bop_extra_files:
            if extra == "tless":
                # https://github.com/kirumang/Pix2Pose#download-pre-trained-weights
                to_dl.append(
                    (
                        "cosypose/bop_datasets/tless/all_target_tless.json",
                        BOP_DS_DIR / "tless",
                    )
                )
                to_symlink.append(
                    (BOP_DS_DIR / "tless/models_eval", BOP_DS_DIR / "tless/models")
                )
            elif extra == "ycbv":
                # Friendly names used with YCB-Video
                to_dl += [
                    (
                        "cosypose/bop_datasets/ycbv/ycbv_friendly_names.txt",
                        BOP_DS_DIR / "ycbv",
                    ),
                    # Offsets between YCB-Video and BOP (extracted from BOP readme)
                    (
                        "cosypose/bop_datasets/ycbv/offsets.txt",
                        BOP_DS_DIR / "ycbv",
                    ),
                    # Evaluation models for YCB-Video (used by other works)
                    (
                        "cosypose/bop_datasets/ycbv/models_original",
                        BOP_DS_DIR / "ycbv",
                    ),
                    # Keyframe definition
                    (
                        "cosypose/bop_datasets/ycbv/keyframe.txt",
                        BOP_DS_DIR / "ycbv",
                    ),
                ]

    if args.urdf_models:
        for model in args.urdf_models:
            to_dl.append(
                (
                    f"cosypose/urdfs/{model}",
                    LOCAL_DATA_DIR / "urdfs",
                )
            )

    if args.ycbv_compat_models:
        to_dl += [
            (
                "cosypose/bop_datasets/ycbv/models_bop-compat",
                BOP_DS_DIR / "ycbv",
            ),
            (
                "cosypose/bop_datasets/ycbv/models_bop-compat_eval",
                BOP_DS_DIR / "ycbv",
            ),
        ]

    if args.ycbv_tests:
        to_dl.append(("ycbv-debug.zip", DOWNLOAD_DIR))
        to_unzip.append((DOWNLOAD_DIR / "ycbv-debug.zip", LOCAL_DATA_DIR / "results"))

    if args.cosypose_models:
        for model in args.cosypose_models:
            to_dl.append(
                (
                    f"cosypose/experiments/{model}",
                    LOCAL_DATA_DIR / "experiments",
                )
            )

    if args.megapose_models:
        to_dl.append(
            (
                "megapose/megapose-models/",
                LOCAL_DATA_DIR / "megapose-models/",
                ["--exclude", ".*epoch.*"],
            )
        )

    if args.detections:
        for detection in args.detections:
            to_dl.append(
                (
                    f"cosypose/saved_detections/{detection}.pkl",
                    LOCAL_DATA_DIR / "saved_detections",
                )
            )

    if args.result_id:
        for result in args.result_id:
            to_dl.append(
                (
                    f"cosypose/results/{result}",
                    LOCAL_DATA_DIR / "results",
                )
            )

    if args.bop_result_id:
        for result in args.bop_result_id:
            to_dl += [
                (
                    f"cosypose/bop_predictions/{result}.csv",
                    LOCAL_DATA_DIR / "bop_predictions",
                ),
                (
                    f"cosypose/bop_eval_outputs/{result}",
                    LOCAL_DATA_DIR / "bop_predictions",
                ),
            ]

    if args.texture_dataset:
        to_dl.append(("cosypose/zip_files/textures.zip", DOWNLOAD_DIR))
        to_unzip.append(
            (DOWNLOAD_DIR / "textures.zip", LOCAL_DATA_DIR / "texture_datasets")
        )

    if args.synt_dataset:
        for dataset in args.synt_dataset:
            to_dl.append((f"cosypose/zip_files/{dataset}.zip", DOWNLOAD_DIR))
            to_unzip.append(
                (DOWNLOAD_DIR / f"{dataset}.zip", LOCAL_DATA_DIR / "synt_datasets")
            )

    if args.example_scenario:
        to_dl += [
            (
                "cosypose/custom_scenarios/example/candidates.csv",
                LOCAL_DATA_DIR / "custom_scenarios/example",
            ),
            (
                "cosypose/custom_scenarios/example/scene_camera.json",
                LOCAL_DATA_DIR / "custom_scenarios/example",
            ),
        ]

    if args.all_bop20_models:
        from happypose.pose_estimators.cosypose.cosypose.bop_config import (
            PBR_COARSE,
            PBR_DETECTORS,
            PBR_REFINER,
            SYNT_REAL_COARSE,
            SYNT_REAL_DETECTORS,
            SYNT_REAL_REFINER,
        )

        for model_dict in (
            PBR_DETECTORS,
            PBR_COARSE,
            PBR_REFINER,
            SYNT_REAL_DETECTORS,
            SYNT_REAL_COARSE,
            SYNT_REAL_REFINER,
        ):
            for model in model_dict.values():
                to_dl.append(
                    (
                        f"cosypose/experiments/{model}",
                        LOCAL_DATA_DIR / "experiments",
                    )
                )

    if args.all_bop20_results:
        from happypose.pose_estimators.cosypose.cosypose.bop_config import (
            PBR_INFERENCE_ID,
            SYNT_REAL_4VIEWS_INFERENCE_ID,
            SYNT_REAL_8VIEWS_INFERENCE_ID,
            SYNT_REAL_ICP_INFERENCE_ID,
            SYNT_REAL_INFERENCE_ID,
        )

        for result_id in (
            PBR_INFERENCE_ID,
            SYNT_REAL_INFERENCE_ID,
            SYNT_REAL_ICP_INFERENCE_ID,
            SYNT_REAL_4VIEWS_INFERENCE_ID,
            SYNT_REAL_8VIEWS_INFERENCE_ID,
        ):
            to_dl.append(
                (
                    f"cosypose/results/{result_id}",
                    LOCAL_DATA_DIR / "results",
                )
            )
    if args.examples:
        for example in args.examples:
            to_dl.append((f"examples/{example}.zip", DOWNLOAD_DIR))
            to_unzip.append(
                (DOWNLOAD_DIR / f"{example}.zip", LOCAL_DATA_DIR / "examples")
            )

    # logger.info(f"{to_dl=}")
    async with DownloadClient(mirror=args.mirror) as dl_client:
        for args in to_dl:
            dl_client.create_task(dl_client.adownload(*args))

    for src, dst in to_symlink:
        os.symlink(src, dst)

    for src, dst in to_unzip:
        zipfile.ZipFile(src).extractall(dst)


class DownloadClient:
    def __init__(self, mirror):
        if mirror in MIRRORS:
            mirror = MIRRORS.pop(mirror)
            self.mirrors = [mirror, *MIRRORS.values()]
        elif mirror:
            self.mirrors = [mirror, *MIRRORS.values()]
        else:
            self.mirrors = MIRRORS.values()
        self.client = httpx.AsyncClient()
        self.task_set = set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def aclose(self):
        while len(self.task_set) > 0:
            # for task in list(self.task_set):
            #    await task
            await asyncio.gather(*list(self.task_set))
        await self.client.aclose()

    def create_task(self, awaitable):
        task = asyncio.create_task(awaitable)
        self.task_set.add(task)
        task.add_done_callback(self.task_set.discard)

    async def adownload(self, download_path, local_path, flags=None):
        if flags is None:
            flags = []
        flags = Flags(flags)

        Path_download = Path(download_path)
        if Path_download.name != local_path.name:
            local_path = local_path / Path_download.name

        if not flags.flags_managing(Path_download.name):  # if the dl_path is --excluded
            return

        for mirror in self.mirrors:
            dl = mirror + download_path
            try:
                await asyncio.sleep(random.uniform(0, 3))
                head = await self.client.head(dl)
            except (httpx.PoolTimeout, httpx.ReadTimeout, httpx.ConnectTimeout):
                continue
            if head.is_success or head.is_redirect:
                download_path = dl
                break
        else:
            err = f"Can't find mirror for {download_path}."
            logger.warning(f"{err} -- retrying soon...")
            await asyncio.sleep(random.uniform(5, 10))
            for mirror in self.mirrors:
                dl = mirror + download_path
                try:
                    await asyncio.sleep(random.uniform(0, 3))
                    head = await self.client.head(dl)
                except (httpx.PoolTimeout, httpx.ReadTimeout, httpx.ConnectTimeout):
                    continue
                if head.is_success or head.is_redirect:
                    download_path = dl
                    break
            else:
                raise ValueError(err)
        if (
            not download_path.endswith("/")
            and not httpx.head(download_path).is_redirect
        ):  # file
            await self.download_file(download_path, local_path)
        else:
            if not download_path.endswith("/"):
                download_path += "/"
            await self.download_dir(download_path, local_path, flags)

    async def download_dir(self, download_path, local_path, flags):
        try:
            await asyncio.sleep(random.uniform(0, 3))
            r = await self.client.get(download_path)
        except (httpx.PoolTimeout, httpx.ReadTimeout, httpx.ConnectTimeout):
            logger.error(f"Failed {download_path} with GET timeout")
            return
        if r.status_code != 200:
            logger.error(f"Failed {download_path} with code {r.status_code}")
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        soup = BeautifulSoup(r.content, "html.parser")
        logger.info(f"Copying {download_path} to {local_path}")

        for link in soup.find_all("a"):
            href: str = link.get("href")
            if any(href.startswith(wrong) for wrong in ["?", ".."]):
                continue
            if not flags.flags_managing(href):
                continue
            if href.endswith("/"):
                self.create_task(
                    self.download_dir(download_path + href, local_path / href, flags),
                )
            else:
                self.create_task(
                    self.download_file(download_path + href, local_path / href),
                )

    async def download_file(self, download_path, local_path):
        local_path = Path(local_path)
        if local_path.exists():
            # logger.info(f"Existing {download_path=}")
            local_size = local_path.stat().st_size
            try:
                await asyncio.sleep(random.uniform(0, 3))
                head = await self.client.head(download_path)
            except (
                httpx.PoolTimeout,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
            ) as e:
                logger.error(f"Failed {download_path} with {e}")
                return
            if "content-length" in head.headers:
                if local_size == int(head.headers["content-length"]):
                    logger.info(f"Skipping {download_path} already fully downloaded")
                    return
                else:
                    logger.info(f"Retrying incomplete {download_path}")
        logger.info(f"Copying {download_path} to {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as f:
            try:
                await asyncio.sleep(random.uniform(5, 10))
                async with self.client.stream("GET", download_path) as r:
                    total = None
                    if "Content-Length" in r.headers:
                        total = int(r.headers["Content-Length"])
                    with tqdm(
                        desc=local_path.name,
                        total=total,
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                    ) as progress:
                        num_bytes_downloaded = r.num_bytes_downloaded
                        async for chunk in r.aiter_bytes():
                            f.write(chunk)
                            progress.update(
                                r.num_bytes_downloaded - num_bytes_downloaded
                            )
                            num_bytes_downloaded = r.num_bytes_downloaded
                if r.status_code != 200:
                    logger.error(f"Failed {download_path} with code {r.status_code}")
                    return
            except (httpx.PoolTimeout, httpx.ReadTimeout, httpx.ConnectTimeout):
                logger.error(f"Failed {download_path} with stream timeout")
                return


class Flags:
    def __init__(self, flags: [str]):
        # only '--exclude' were used before so this is the only flag currently usable
        # if you need to use other flags, feel free to implement them here
        self.exclude_set: Set[str] = set()

        parser = argparse.ArgumentParser("Flags parsing")
        parser.add_argument("--exclude", default="", type=str)
        args = parser.parse_args(flags)

        if args.exclude:
            self.exclude_set.add(args.exclude)

    def flags_managing(flags, href):
        for el in flags.exclude_set:
            if re.fullmatch(el, href):
                return False
        return True


if __name__ == "__main__":
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    asyncio.run(main())
