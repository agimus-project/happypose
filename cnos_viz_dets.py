import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

MEGAPOSE_DATA_DIR = Path(os.environ.get('MEGAPOSE_DATA_DIR'))
CNOS_SUBMISSION_DIR = Path(os.environ.get('CNOS_SUBMISSION_DIR'))

# CNOS_SUBMISSION_FILES = {
#     "ycbv": 'baseline-sam-dinov2-blenderproc4bop_ycbv-test_a491e9fe-1137-4585-9c80-0a2056a3eb9c.json',
#     "lmo": 'baseline-sam-dinov2-blenderproc4bop_lmo-test_2f321533-59ae-4541-b65e-6b4e4fb9d391.json',
#     "tless": 'baseline-sam-dinov2-blenderproc4bop_tless-test_3305b238-3d93-4954-81ba-3ff3786265d9.json',
#     "tudl": 'baseline-sam-dinov2-blenderproc4bop_tudl-test_c6cd05c1-89a1-4fe5-88b9-c1b57ef15694.json',
#     "icbin": 'baseline-sam-dinov2-blenderproc4bop_icbin-test_f58b6868-7e70-4ab2-9332-65220849f8c1.json',
#     # "itodd": 'baseline-sam-dinov2-blenderproc4bop_itodd-test_82442e08-1e79-4f54-8e88-7ad6b986dd96.json',
#     # "hb": 'baseline-sam-dinov2-blenderproc4bop_hb-test_f32286f9-05f5-4123-862f-18f00e67e685.json',
# }


CNOS_SUBMISSION_FILES = {
    "ycbv": 'fastSAM_pbr_ycbv.json', 
    "lmo": 'fastSAM_pbr_lmo.json', 
    "tless": 'fastSAM_pbr_tless.json', 
    "tudl": 'fastSAM_pbr_tudl.json', 
    "icbin": 'fastSAM_pbr_icbin.json', 
    "itodd": 'fastSAM_pbr_itodd.json', 
    "hb": 'fastSAM_pbr_hb.json', 
}


TEST_DIRS = {
    'ycbv': 'test',
    'lmo': 'test',
    'tless': 'test_primesense',
    'tudl': 'test',
    'icbin': 'test',
    'itodd': 'test',
    'hb': 'test_primesense',
}

IMG_DIRS = {
    'ycbv': 'rgb',
    'lmo': 'rgb',
    'tless': 'rgb',
    'tudl': 'rgb',
    'icbin': 'rgb',
    'itodd': 'gray',
    'hb': 'rgb',
}


SET = 'test'

# DS_NAME = 'itodd'
# SCENE_ID = 1
# VIEW_ID = 10

# DS_NAME = 'ycbv'
# SCENE_ID = 53
# VIEW_ID = 14

DS_NAME = 'icbin'
SCENE_ID = 2
VIEW_ID = 50




TMP_DIR = Path('tmp_cnos/')
TMP_DIR.mkdir(exist_ok=True)

ds_bop_dir = MEGAPOSE_DATA_DIR / f'bop_datasets/{DS_NAME}'

cnos_submission_path = CNOS_SUBMISSION_DIR / CNOS_SUBMISSION_FILES[DS_NAME]


assert(ds_bop_dir.exists())
assert(cnos_submission_path.exists())


# Get nb of 
ds_bop_dir = MEGAPOSE_DATA_DIR / f'bop_datasets/{DS_NAME}'
test_targets_path = ds_bop_dir / 'test_targets_bop19.json'
test_targets = json.loads(test_targets_path.read_text())
df_test_targets = pd.DataFrame.from_records(test_targets)

df_targets = df_test_targets[(df_test_targets['scene_id'] == SCENE_ID) & ((df_test_targets['im_id'] == VIEW_ID))]
nb_det_gt = df_targets.inst_count.sum()


###############################
# Retrieve and check directories
set_dir = ds_bop_dir / TEST_DIRS[DS_NAME]
# available scenes
scenes = sorted(list(int(d.name) for d in set_dir.iterdir()))
if SCENE_ID not in scenes:
    raise ValueError(f'Not good SCENE_ID: {SCENE_ID} not in {scenes}')

scene_dir = set_dir / '{:>06}'.format(SCENE_ID)
imgs_dir = scene_dir / IMG_DIRS[DS_NAME]
imgs_path = sorted(list(d for d in imgs_dir.iterdir()))
file_format = imgs_path[0].suffix
# available views
view_ids = [int(p.stem) for p in imgs_path]
if VIEW_ID not in view_ids:
    raise ValueError(f'Not good VIEW_ID: {VIEW_ID} not in {view_ids}')




###############################
THRESHOLD_SCORE = 0.3


def filter_detections(df_all_det, scene_id, view_id):
    return df_all_det[(df_all_det['scene_id'] == scene_id) & (df_all_det['image_id'] == view_id)]

def bbox_format(bboxes_xywh):
    # Bounding box formats:
    # - CNOS/SAM baseline.json: [xmin, ymin, width, height]
    # - Megapose expects: [xmin, ymin, xmax, ymax]
    bboxes_xyxy = []
    for bb in bboxes_xywh:
        x, y, w, h = bb
        bboxes_xyxy.append([float(v) for v in [x, y, x+w, y+h]])

    return bboxes_xyxy

def score2color(score, threshold):
    # color: BGR format
    if score > threshold:
        # v = int(score*255)
        v = 255
        return 0, v, v
    else:
        return 255, 0, 0


def score2rankids(scores):
    # create ranks ids sorted as the score in descending order 
    return [r for _, r in sorted(zip(scores, list(range(len(scores)))), reverse=True)]


def draw_bboxes(img, bboxes_xywh, categories, scores=0.8, nb_det_gt=np.inf, thresh=0.7):
    if isinstance(scores, float):
        scores = len(bboxes_xywh)*[scores]

    assert(len(bboxes_xywh) == len(scores))

    ranks_ids = score2rankids(scores)

    # iterate in the order of the ids
    for rank, rid in enumerate(ranks_ids):
        # show only 
        if rank > nb_det_gt:
            break
        score = round(scores[rid],4)
        cat = int(categories[rid])
        bb = bboxes_xywh[rid]
        color = score2color(score, thresh)

        # print(f'\n{rid}')
        # print('rank, score, cat, bb')
        # print(rank, score, cat, bb)

        # Bounding box formats:
        # - CNOS/SAM baseline.json: [xmin, ymin, width, height]
        x, y, w, h = bb
        start_point = x, y
        end_point = x+w, y+h
        thickness_rect = 1
        img = cv2.rectangle(img, start_point, end_point, color, thickness_rect)

        # fontScale
        fontScale = 0.5
        thickness_txt = 2
        shift_x, shift_y = 4, 12
        # txt = f'{rank}_{cat}_{score}'
        txt = f'{rank}'
        img = cv2.putText(img, txt, (x+shift_x, y+shift_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale, color, thickness_txt, cv2.LINE_AA)

    return img

###########
# Retrieve image
rgb_path = imgs_dir / '{:>06}{}'.format(VIEW_ID, file_format)
img = cv2.imread(rgb_path.as_posix())

###########
# Retrieve detections
# As in prediction_runner.py (variable called object_data)
det_dic = json.loads(cnos_submission_path.read_text())
df_all_det = pd.DataFrame.from_records(det_dic)
df_all_det = df_all_det.sort_values(['scene_id', 'image_id'])

# Remove low scores
# df_all_det = df_all_det[df_all_det['score'] > THRESHOLD_SCORE]

df_det = filter_detections(df_all_det, SCENE_ID, VIEW_ID)
print('NB detections: ', len(df_det))
print('NB GT detections: ', nb_det_gt)

categories = df_det['category_id'].to_list()
bboxes_xywh = df_det['bbox'].to_list()
scores = df_det['score'].to_list()


# N_EXTRA_GT_DET = 0
N_EXTRA_GT_DET = 20
img = draw_bboxes(img, bboxes_xywh, categories, scores)  # SHOW ALL DETS
# img = draw_bboxes(img, bboxes_xywh, categories, scores, nb_det_gt=nb_det_gt+N_EXTRA_GT_DET)  # SHOW BEST DETS

img_out_name = f'cnos_dets_viz_{DS_NAME}_{SCENE_ID}_{VIEW_ID}.png'
img_out_path = TMP_DIR / img_out_name
print(f'Saving {img_out_path.as_posix()}')
cv2.imwrite(img_out_path.as_posix(), img)