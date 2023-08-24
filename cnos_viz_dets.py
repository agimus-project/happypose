import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2

MEGAPOSE_DATA_DIR = Path(os.environ.get('MEGAPOSE_DATA_DIR'))
CNOS_SUBMISSION_DIR = Path(os.environ.get('CNOS_SUBMISSION_DIR'))

CNOS_SUBMISSION_FILES = {
    "ycbv": 'cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json', 
    "lmo": 'cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json', 
    "tless": 'cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json', 
    "tudl": 'cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json', 
    "icbin": 'cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json', 
    "itodd": 'cnos-fastsam_itodd-test_df32d45b-301c-4fc9-8769-797904dd9325.json', 
    "hb": 'cnos-fastsam_hb-test_db836947-020a-45bd-8ec5-c95560b68011.json', 
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

# DS_NAME = 'icbin'
# SCENE_ID = 2
# VIEW_ID = 11

# DS_NAME = 'tless'
# SCENE_ID = 18
# VIEW_ID = 452

DS_NAME = 'tudl'
SCENE_ID = 3
VIEW_ID = 4875


LOCALIZATION6D = True




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
obj_ids = df_targets.obj_id.to_list()
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
        # show only first dets
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

# 6D localization -> filter out objects that are not known to be in the image
if LOCALIZATION6D:
    print(len(df_det))
    df_det = df_det[df_det['category_id'].isin(obj_ids)]
    print(len(df_det))

categories = df_det['category_id'].to_list()
bboxes_xywh = df_det['bbox'].to_list()
scores = df_det['score'].to_list()


N_EXTRA_GT_DET = 0
# N_EXTRA_GT_DET = 20
# img = draw_bboxes(img, bboxes_xywh, categories, scores)  # SHOW ALL DETS
img = draw_bboxes(img, bboxes_xywh, categories, scores, nb_det_gt=nb_det_gt+N_EXTRA_GT_DET)  # SHOW BEST DETS

if LOCALIZATION6D:
    img_out_name = f'cnos_dets_viz_{DS_NAME}_{SCENE_ID}_{VIEW_ID}_filt.png'
else:
    img_out_name = f'cnos_dets_viz_{DS_NAME}_{SCENE_ID}_{VIEW_ID}.png'
img_out_path = TMP_DIR / img_out_name
print(f'Saving {img_out_path.as_posix()}')
cv2.imwrite(img_out_path.as_posix(), img)