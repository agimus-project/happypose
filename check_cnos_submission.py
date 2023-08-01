import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


MEGAPOSE_DATA_DIR = Path(os.environ.get('MEGAPOSE_DATA_DIR'))
CNOS_SUBMISSION_DIR = Path(os.environ.get('CNOS_SUBMISSION_DIR'))

CNOS_SUBMISSION_FILES = {
    "ycbv": 'baseline-sam-dinov2-blenderproc4bop_ycbv-test_a491e9fe-1137-4585-9c80-0a2056a3eb9c.json',
    "lmo": 'baseline-sam-dinov2-blenderproc4bop_lmo-test_2f321533-59ae-4541-b65e-6b4e4fb9d391.json',
    "tless": 'baseline-sam-dinov2-blenderproc4bop_tless-test_3305b238-3d93-4954-81ba-3ff3786265d9.json',
    "tudl": 'baseline-sam-dinov2-blenderproc4bop_tudl-test_c6cd05c1-89a1-4fe5-88b9-c1b57ef15694.json',
    "icbin": 'baseline-sam-dinov2-blenderproc4bop_icbin-test_f58b6868-7e70-4ab2-9332-65220849f8c1.json',
    "itodd": 'baseline-sam-dinov2-blenderproc4bop_itodd-test_82442e08-1e79-4f54-8e88-7ad6b986dd96.json',
    "hb": 'baseline-sam-dinov2-blenderproc4bop_hb-test_f32286f9-05f5-4123-862f-18f00e67e685.json',
}

SHOW = False

for ds_name in CNOS_SUBMISSION_FILES:

    detections_path = CNOS_SUBMISSION_DIR / CNOS_SUBMISSION_FILES[ds_name]

    # Test sub dataset

    # # BOP
    ds_bop_dir = MEGAPOSE_DATA_DIR / f'bop_datasets/{ds_name}'
    print(ds_bop_dir)
    ds_bop_dir = Path(ds_bop_dir)
    test_targets_path = ds_bop_dir / 'test_targets_bop19.json'

    if not ds_bop_dir.exists():
        print(f'missing dataset {ds_name}')
        continue
    # assert(ds_bop_dir.exists())
    # assert(detections_path.exists())
    # assert(test_targets_path.exists())


    #################
    # Get all image ids for each scene id from ds test
    #################
    test_dir = ds_bop_dir / 'test'
    all_scene_ids = [d.name for d in test_dir.iterdir()]

    test_targets = json.loads(test_targets_path.read_text())
    df_test_targets = pd.DataFrame.from_records(test_targets)


    scene_image_ids = {}
    scene_nb_obj = {}
    for sid_str in all_scene_ids:

        rgb_dir = test_dir / sid_str / Path('rgb')

        sid = int(sid_str)

        # retrieve submission subtest
        img_ids_subtest = df_test_targets[df_test_targets['scene_id'] == sid]['im_id'].unique()

        # retrieve dataset image ids and filter them
        all_img_ids_for_sid = sorted([int(f.stem) for f in rgb_dir.glob('*')])
        scene_image_ids[sid] = [iid for iid in all_img_ids_for_sid if iid in img_ids_subtest]
        # scene_image_ids[sid] = all_img_ids_for_sid


    #################
    # Get all image ids with detections for each scene id from ds test
    #################
    # As in prediction_runner.py (variable called object_data)
    # """
    # In [3]: list(dets[0].keys())
    # Out[3]: 
    # ['scene_id',
    #  'image_id',
    #  'category_id',
    #  'bbox',
    #  'score',
    #  'time',
    #  'segmentation']
    # """
    all_dets = json.loads(detections_path.read_text())
    df_all_dets = pd.DataFrame.from_records(all_dets)

    # print('scene ids:', df_all_dets['scene_id'].unique())
    # print('nb_scene_id: ', df_all_dets['scene_id'].nunique())
    # print('nb_image_id: ', df_all_dets['image_id'].nunique())
    # print('nb_category_id: ', df_all_dets['category_id'].nunique())

    df_unique_det = df_all_dets.groupby('scene_id')['image_id'].nunique()

    df_image_id = df_all_dets.groupby('scene_id')['image_id'].unique()
    scene_image_ids_cnos = df_image_id.to_dict()


    ### NB unique images vs nb images with detection
    ycbv_nb_images = sum(len(ids) for ids in scene_image_ids.values())
    cnos_nb_images_with_det = sum(len(ids) for ids in scene_image_ids_cnos.values())

    print('ycbv_nb_images: ', ycbv_nb_images)
    print('cnos_nb_images_with_det: ', cnos_nb_images_with_det)
    print('% img with CNOS detection: ', 100*(cnos_nb_images_with_det/ycbv_nb_images))

    # which submissions are missing??
    d_missing = {}
    for sid in scene_image_ids:
        img_ids_set_bop = set(scene_image_ids[sid])
        img_ids_set_cnos = set(scene_image_ids_cnos[sid])
        missing_ids = img_ids_set_bop - img_ids_set_cnos
        d_missing[sid] = sorted(list(img_ids_set_bop - img_ids_set_cnos))

    print('Missing images for each scene')
    print(d_missing)



    ######################
    ## How many detections per image?
    ######################

    cat_ids_per_sid_iid = df_all_dets.groupby(['scene_id', 'image_id']).category_id.nunique()
    all_obj_nb = cat_ids_per_sid_iid.to_list()

    plt.figure('Detection numbers')
    plt.title(f'Number of detections per image {ds_name}')
    plt.boxplot(all_obj_nb)



    scene_ids = list(scene_image_ids_cnos.keys())
    markersize = 2
    plt.figure()
    for sid in scene_ids:
        plt.plot(len(scene_image_ids[sid])*[sid], scene_image_ids[sid], 'xr', label=f'{ds_name} gt', markersize=markersize)
        plt.plot(len(scene_image_ids_cnos[sid])*[sid], scene_image_ids_cnos[sid], 'xb', label=f'{ds_name} cnos', markersize=markersize)


    plt.title(f'CNOS submission on {ds_name}, Red -> missing detection')
    plt.xlabel('Scene id')
    plt.ylabel('Image ids')
    # plt.legend()
    fig_name = f'cnos_dets_{ds_name}'
    print('Saving ', fig_name)
    plt.savefig(fig_name)

if SHOW:
    plt.show()
