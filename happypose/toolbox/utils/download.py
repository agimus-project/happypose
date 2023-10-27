import argparse
import logging
import os
import subprocess
import zipfile
from pathlib import Path

import wget

from happypose.pose_estimators.cosypose.cosypose.config import (
    BOP_DS_DIR,
    LOCAL_DATA_DIR,
    PROJECT_DIR,
)
from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger

logger = get_logger(__name__)

RCLONE_CFG_PATH = (PROJECT_DIR / 'rclone.conf')
RCLONE_ROOT = 'happypose:'
DOWNLOAD_DIR = LOCAL_DATA_DIR / 'downloads'
DOWNLOAD_DIR.mkdir(exist_ok=True)
BOP_SRC = 'https://bop.felk.cvut.cz/media/data/bop_datasets/'
BOP_DATASETS = {
    'ycbv': {
        'splits': ['train_real', 'train_synt', 'test_all']
    },

    'tless': {
        'splits': ['test_primesense_all', 'train_primesense'],
    },

    'hb': {
        'splits': ['test_primesense_all', 'val_primesense'],
    },

    'icbin': {
        'splits': ['test_all'],
    },

    'itodd': {
        'splits': ['val', 'test_all'],
    },

    'lm': {
        'splits': ['test_all'],
    },

    'lmo': {
        'splits': ['test_all'],
        'has_pbr': False,
    },

    'tudl': {
        'splits': ['test_all', 'train_real']
    },
}

BOP_DS_NAMES = list(BOP_DATASETS.keys())


def main():
    parser = argparse.ArgumentParser('CosyPose download utility')
    parser.add_argument('--bop_dataset', default='', type=str, choices=BOP_DS_NAMES)
    parser.add_argument('--bop_src', default='bop', type=str, choices=['bop', 'gdrive'])
    parser.add_argument('--bop_extra_files', default='', type=str, choices=['ycbv', 'tless'])
    parser.add_argument('--cosypose_models', default='', type=str)
    parser.add_argument("--megapose_models", action="store_true")
    parser.add_argument('--urdf_models', default='', type=str)
    parser.add_argument('--ycbv_compat_models', action='store_true')
    parser.add_argument('--texture_dataset', action='store_true')
    parser.add_argument('--result_id', default='', type=str)
    parser.add_argument('--bop_result_id', default='', type=str)
    parser.add_argument('--synt_dataset', default='', type=str)
    parser.add_argument('--detections', default='', type=str)
    parser.add_argument('--example_scenario', action='store_true')
    parser.add_argument('--pbr_training_images', action='store_true')
    parser.add_argument('--all_bop20_results', action='store_true')
    parser.add_argument('--all_bop20_models', action='store_true')

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.bop_dataset:
        if args.bop_src == 'bop':
            download_bop_original(args.bop_dataset, args.pbr_training_images and BOP_DATASETS[args.bop_dataset].get('has_pbr', True))
        elif args.bop_src == 'gdrive':
            download_bop_gdrive(args.bop_dataset)

    if args.bop_extra_files:
        if args.bop_extra_files == 'tless':
            # https://github.com/kirumang/Pix2Pose#download-pre-trained-weights
            download(f'cosypose/bop_datasets/tless/all_target_tless.json', BOP_DS_DIR / 'tless')
            os.symlink(BOP_DS_DIR / 'tless/models_eval', BOP_DS_DIR / 'tless/models')
        elif args.bop_extra_files == 'ycbv':
            # Friendly names used with YCB-Video
            download(f'cosypose/bop_datasets/ycbv/ycbv_friendly_names.txt', BOP_DS_DIR / 'ycbv')
            # Offsets between YCB-Video and BOP (extracted from BOP readme)
            download(f'cosypose/bop_datasets/ycbv/offsets.txt', BOP_DS_DIR / 'ycbv')
            # Evaluation models for YCB-Video (used by other works)
            download(f'cosypose/bop_datasets/ycbv/models_original', BOP_DS_DIR / 'ycbv')
            # Keyframe definition
            download(f'cosypose/bop_datasets/ycbv/keyframe.txt', BOP_DS_DIR / 'ycbv')

    if args.urdf_models:
        download(f'cosypose/urdfs/{args.urdf_models}', LOCAL_DATA_DIR / 'urdfs')

    if args.ycbv_compat_models:
        download(f'cosypose/bop_datasets/ycbv/models_bop-compat', BOP_DS_DIR / 'ycbv')
        download(f'cosypose/bop_datasets/ycbv/models_bop-compat_eval', BOP_DS_DIR / 'ycbv')

    if args.cosypose_models:
        download(f'cosypose/experiments/{args.cosypose_models}', LOCAL_DATA_DIR / 'experiments')
    
    if args.megapose_models:
        # rclone copyto inria_data:megapose-models/ megapose-models/
        #     --exclude="**epoch**" --config $HAPPYPOSE_DATA_DIR/rclone.conf -P
        download(
            f"megapose/megapose-models/",
            LOCAL_DATA_DIR / "megapose-models/",
            flags=["--exclude", "*epoch*"],
        )

    if args.detections:
        download(f'cosypose/saved_detections/{args.detections}.pkl', LOCAL_DATA_DIR / 'saved_detections')

    if args.result_id:
        download(f'cosypose/results/{args.result_id}', LOCAL_DATA_DIR / 'results')

    if args.bop_result_id:
        csv_name = args.bop_result_id + '.csv'
        download(f'cosypose/bop_predictions/{csv_name}', LOCAL_DATA_DIR / 'bop_predictions')
        download(f'cosypose/bop_eval_outputs/{args.bop_result_id}', LOCAL_DATA_DIR / 'bop_predictions')

    if args.texture_dataset:
        download('cosypose/zip_files/textures.zip', DOWNLOAD_DIR)
        logger.info('Extracting textures ...')
        zipfile.ZipFile(DOWNLOAD_DIR / 'textures.zip').extractall(LOCAL_DATA_DIR / 'texture_datasets')

    if args.synt_dataset:
        zip_name = f'{args.synt_dataset}.zip'
        download(f'cosypose/zip_files/{zip_name}', DOWNLOAD_DIR)
        logger.info('Extracting textures ...')
        zipfile.ZipFile(DOWNLOAD_DIR / zip_name).extractall(LOCAL_DATA_DIR / 'synt_datasets')

    if args.example_scenario:
        download(f'cosypose/custom_scenarios/example/candidates.csv', LOCAL_DATA_DIR / 'custom_scenarios/example')
        download(f'cosypose/custom_scenarios/example/scene_camera.json', LOCAL_DATA_DIR / 'custom_scenarios/example')

    if args.all_bop20_models:
        from happypose.pose_estimators.cosypose.cosypose.bop_config import (
            PBR_COARSE,
            PBR_DETECTORS,
            PBR_REFINER,
            SYNT_REAL_COARSE,
            SYNT_REAL_DETECTORS,
            SYNT_REAL_REFINER,
        )
        for model_dict in (PBR_DETECTORS, PBR_COARSE, PBR_REFINER,
                           SYNT_REAL_DETECTORS, SYNT_REAL_COARSE, SYNT_REAL_REFINER):
            for model in model_dict.values():
                download(f'cosypose/experiments/{model}', LOCAL_DATA_DIR / 'experiments')

    if args.all_bop20_results:
        from happypose.pose_estimators.cosypose.cosypose.bop_config import (
            PBR_INFERENCE_ID,
            SYNT_REAL_4VIEWS_INFERENCE_ID,
            SYNT_REAL_8VIEWS_INFERENCE_ID,
            SYNT_REAL_ICP_INFERENCE_ID,
            SYNT_REAL_INFERENCE_ID,
        )
        for result_id in (PBR_INFERENCE_ID, SYNT_REAL_INFERENCE_ID, SYNT_REAL_ICP_INFERENCE_ID,
                          SYNT_REAL_4VIEWS_INFERENCE_ID, SYNT_REAL_8VIEWS_INFERENCE_ID):
            download(f'cosypose/results/{result_id}', LOCAL_DATA_DIR / 'results')


def run_rclone(cmd, args, flags):
    rclone_cmd = ['rclone', cmd] + args + flags + ['--config', str(RCLONE_CFG_PATH)]
    logger.debug(' '.join(rclone_cmd))
    print(rclone_cmd)
    subprocess.run(rclone_cmd)


def download(download_path, local_path, flags=[]):
    download_path = Path(download_path)
    if download_path.name != local_path.name:
        local_path = local_path / download_path.name
    if '.' in str(download_path):
        rclone_path = RCLONE_ROOT + str(download_path)
    else:
        rclone_path = RCLONE_ROOT + str(download_path) + "/"
    local_path = str(local_path)
    logger.info(f"Copying {rclone_path} to {local_path}")
    run_rclone("copyto", [rclone_path, local_path], flags=flags + ["-P"])


def download_bop_original(ds_name, download_pbr):
    filename = f'{ds_name}_base.zip'
    wget_download_and_extract(BOP_SRC + filename, BOP_DS_DIR)

    suffixes = ['models'] + BOP_DATASETS[ds_name]['splits']
    if download_pbr:
        suffixes += ['train_pbr']
    for suffix in suffixes:
        wget_download_and_extract(BOP_SRC + f'{ds_name}_{suffix}.zip', BOP_DS_DIR / ds_name)


def download_bop_gdrive(ds_name):
    download(f'bop_datasets/{ds_name}', BOP_DS_DIR / ds_name)


def wget_download_and_extract(url, out):
    tmp_path = DOWNLOAD_DIR / url.split('/')[-1]
    if tmp_path.exists():
        logger.info(f'{url} already downloaded: {tmp_path}...')
    else:
        logger.info(f'Download {url} at {tmp_path}...')
        wget.download(url, out=tmp_path.as_posix())
    logger.info(f'Extracting {tmp_path} at {out}.')
    zipfile.ZipFile(tmp_path).extractall(out)


if __name__ == '__main__':
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    main()
