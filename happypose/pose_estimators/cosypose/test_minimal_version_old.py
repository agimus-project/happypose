import numpy as np
import cv2
import time
from pathlib import Path

import torch
from cosypose_wrapper import CosyPoseWrapper
from happypose.pose_estimators.cosypose.cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from happypose.pose_estimators.cosypose.cosypose.visualization.singleview import render_prediction_wrt_camera
import cosypose
from torchvision.ops import box_iou
from torch.testing import assert_close

"""
First, we set the variables we want to test with
"""

dataset_to_use = 'ycbv'  # tless or ycbv

IMG_RES = 640, 480 
# Realsense 453i intrinsics (from rostopic camera_info)
K_rs = np.array([615.1529541015625, 0.0, 324.5750732421875, 
    0.0, 615.2452392578125, 237.81765747070312, 
    0.0, 0.0, 1.0]).reshape((3,3))

img_dir = 'imgs'
# image_name = 'all_Color.png'
# image_name = 'all_far_Color.png'
# image_name = 'banana_Color.png'
image_name = 'cheezit_Color.png'
# image_name = 'wood_block_Color.png'

# imread stores color dim in the BGR order by default
brg = cv2.imread(img_dir + '/' + image_name)
# CosyPose uses a RGB representation internally?
rgb = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)
cosy_wrapper = CosyPoseWrapper(dataset_name=dataset_to_use, n_workers=8)
preds = cosy_wrapper.inference(rgb, K_rs)

# rendering
renderer = BulletSceneRenderer('ycbv', gpu_renderer=False)
cam = {
    'resolution': IMG_RES,
    'K': K_rs,
    'TWC': np.eye(4),
}
rgb_render = render_prediction_wrt_camera(renderer, preds, cam)
mask = ~(rgb_render.sum(axis=-1) == 0)

alpha = 0.1

rgb_n_render = rgb.copy()
rgb_n_render[mask] = rgb_render[mask]

# make the image background a bit fairer than the render
rgb_overlay = np.zeros_like(rgb_render)
rgb_overlay[~mask] = rgb[~mask] * 0.6 + 255 * 0.4
rgb_overlay[mask] = rgb_render[mask] * 0.8 + 255 * 0.2

# Test the initialization of the detection and pose predictor models
def test_init_wrapper():
    assert(type(cosy_wrapper.detector) == cosypose.integrated.detector.Detector)
    assert(type(cosy_wrapper.pose_predictor) == cosypose.integrated.pose_predictor.CoarseRefinePosePredictor)
    assert(cosy_wrapper.dataset_name == dataset_to_use)

# Test that the number of detected objects are the same, and the score is similar
# Test that the number of predictions and the inference time corresponds to the original one
def test_inference_integration():
    t = time.time()
    preds = cosy_wrapper.inference(rgb, K_rs)
    #assert(time.time() - t < 5)
    print("time = ", time.time() - t)
    assert(len(preds) == 1)

##################################
##################################
##################################
#
#   Neural nets predictions
#
##################################
##################################
##################################


# Test that the score and the bounding boxes are the same
def test_inference_detection():
    image = rgb

    images = torch.from_numpy(image).float().unsqueeze_(0)
    images = images.permute(0, 3, 1, 2) / 255
    
    
    box_detections = cosy_wrapper.detector.get_detections(images=images, one_instance_per_class=False,
                                                      # detection_th=0.8, output_masks=False, mask_th=0.9)
                                                      detection_th=0.7, output_masks=False, mask_th=0.8)

    assert_close(box_detections.bboxes, torch.tensor([[218.9421, 121.9166, 415.8912, 381.7610]]))
    assert(box_detections.infos['score'][0] > 0.9999)


# Test that the output of the pose detection models are the same

# The results obtained here differ of the results obtained by running the scrip
# minimal_test.py. It is to be investigated to know why.
def test_inference(ndarrays_regression):
    dic_tensors = dict()
    dic_tensors['poses'] = preds.poses.detach().cpu().numpy()
    dic_tensors['poses_input'] = preds.poses_input.detach().cpu().numpy()
    dic_tensors['K_crop'] = preds.K_crop.detach().cpu().numpy()
    dic_tensors['boxes_rend'] = preds.boxes_rend.detach().cpu().numpy()
    dic_tensors['boxes_crop'] = preds.boxes_crop.detach().cpu().numpy()
    # Setting tolerance to arbitrary level
    ndarrays_regression.check(dic_tensors, default_tolerance=dict(atol=1e-8, rtol=1e-8))   
    
    # Saving old version
    """
    assert_close(preds.poses, 
                 torch.tensor([[[0.116784,  0.993015,  0.016808, -0.009588],
                            [ 0.189639, -0.005684, -0.981837,  0.012246],
                            [-0.974884,  0.117850, -0.188978,  0.523082],
                            [ 0.000000,  0.000000,  0.000000,  1.000000]]]))

    assert_close(preds.poses_input,
                 torch.tensor([[[ 0.154123,  0.987574,  0.030725, -0.009887],
                            [ 0.247309, -0.008452, -0.968900,  0.011040],
                            [-0.956600,  0.156928, -0.245539,  0.523046],
                            [ 0.000000,  0.000000,  0.000000,  1.000000]]]))
    assert_close(preds.K_crop,
                 torch.tensor([[[398.229767,   0.000000, 167.027328],
                            [  0.000000, 398.289490, 111.092880],
                            [  0.000000,   0.000000,   1.000000]]]))
    assert_close(preds.boxes_rend,
                 torch.tensor([[214.328354, 120.109589, 419.351959, 383.208740]]))
    assert_close(preds.boxes_crop,
                 torch.tensor([[ 65.792450,  65.438034, 560.102417, 436.170532]]))
    """
    
    
    
##################################
##################################
##################################
#
#   Rendering
#
##################################
##################################
##################################


# Compare the results of the render prediction 
def test_render_data(ndarrays_regression):
    dic_arrays = dict()
    dic_arrays['rgb_render'] = rgb_render
    dic_arrays['mask'] = mask
    dic_arrays['rgb_n_render'] = rgb_n_render
    dic_arrays['rgb_overlay'] = rgb_overlay
    ndarrays_regression.check(dic_arrays)
    
    
##################################
###########   render_prediction_wrt_camera
##################################
 
# Objectif de la fonction : Tester que la liste d'objets créée est bien la même

# Vérifie que l'objet détecté est bien le bon, ainsi que la position à laquelle il est détecté est la même
def test_render_predicton_wrt_camera_objects(ndarrays_regression):
    
    list_objects = []
    pred = preds
    camera = cam
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
        )
        list_objects.append(obj)
    for dic in list_objects:
        ndarrays_regression.check(dic)
    """
    # Content:
    [{'name': 'obj_000002', 'color': (1, 1, 1, 1), 'TWO': array([[ 0.11678387,  0.9930151 ,  0.01680777, -0.00958838],
       [ 0.18963857, -0.00568392, -0.98183745,  0.01224616],
       [-0.97488403,  0.1178502 , -0.18897778,  0.5230816 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],
      dtype=float32)}]
    """


# Objectif de la fonction : Tester que le rgb_render retourné est bien le même
# Ici nous vérifions que l'image issue du renderer est la même
def test_render_predicton_wrt_camera(ndarrays_regression):
    
    rgb_render = render_prediction_wrt_camera(renderer, preds, cam)
    dic_render = {'render': rgb_render}
    ndarrays_regression.check(dic_render)



##################################
###########   BulletSceneRender : render_scene
##################################



from happypose.pose_estimators.cosypose.cosypose.lib3d import Transform
import pybullet as pb

# Objectif : Tester individuellement que setup_scene et 
# render_images retournent le bon élément

# Test que le get body est bien fonctionnel
# Ici, le body de l'objet détecté est récupéré dans la liste
def test_setup_scene_get_bodies(data_regression, ndarrays_regression):
    
    # Re run of previous parts to access variables
    list_objects = []
    pred = preds
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
    )
    list_objects.append(obj)    
    
    ####### Actual beginning of the test
        
    # variable name change
    obj_infos = list_objects
    
    # Name of the object in the dataset
    # ex : obj_000002
    labels = [obj['name'] for obj in obj_infos]
    
    assert labels == ['obj_000002']
    
    # Exemple urdf dataset:
    #  print("urdf_ds = ", renderer.urdf_ds.index)
    # urdf_ds =           label                                          urdf_path  scale
    #0   obj_000013  /home/emaitre/cosypose/local_data/urdfs/ycbv/o...  0.001

    
    # Load les body des objets qui ne sont pas en cache
    # Pour load un body : on va chercher son index, son path, et on load l'urdf correspondant
    # en utilisant pybullet.loadurdf
    # on ajoute ensuite à self.cache le body qui a été load, correspondant au label
    bodies = renderer.body_cache.get_bodies_by_labels(labels)
    for body in bodies:
        dic_body = body.get_state()
        
        ndarrays_regression.check({'TWO':dic_body.pop('TWO')})
        data_regression.check(dic_body)  


# Test que la modification du body est la bonne (modification du TWO)

### Que fait vraiment cette fonction ? Ce qui est retourné n'est jamais récupéré,
### et n'est pas utilisé ????
# Checker change visual shape ?
def test_setup_scene_modification_bodies(data_regression, ndarrays_regression):
    
    # Re run of previous parts to access variables
    list_objects = []
    pred = preds
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(
            name=row.label,
            color=(1, 1, 1, 1),
            TWO=pred.poses[n].numpy(),
    )
    list_objects.append(obj)        

    # variable name change
    obj_infos = list_objects

    # Name of the object in the dataset
    # ex : obj_000002
    labels = [obj['name'] for obj in obj_infos]
    bodies = renderer.body_cache.get_bodies_by_labels(labels)

    ####### Actual beginning of the test
    

    #ndarrays_regression.check(obj_infos)
    # Récupère la transformation à faire à l'objet pour l'obtenir à partir de la
    # pose détectée ?
    # --> La pose est effectivement modifiée
    for (obj_info, body) in zip(obj_infos, bodies):
        TWO = Transform(obj_info['TWO'])
        body.pose = TWO
        color = obj_info.get('color', None)
        
        # Modifie la couleur de l'objet s'il y a lieu
        if color is not None:
            pb.changeVisualShape(body.body_id, -1, physicsClientId=0, rgbaColor=color)

    for body in bodies:
        dic_body = body.get_state()
        ndarrays_regression.check({'TWO':dic_body.pop('TWO')})
        data_regression.check(dic_body)  

    # Ici, les bodies ont été mis à jour conformément à la pose qui a été détectée ?
    # La mise à jour n'est pas faite dans le cache du renderer d'après le test ?
    # Pourtant, cet élément est bien essentiel et a bien un effet
    
    
##################################
###########   General test
##################################
    

# Redondant avec le test complet ?
def test_render_images():
    
    from happypose.pose_estimators.cosypose.cosypose.simulator.camera import Camera
    render_depth = False
    cam = {
    'resolution': IMG_RES,
    'K': K_rs,
    'TWC': np.eye(4),
    }

    # variable settings
    camera = cam
    cam_infos = [camera]
    
    # beginning of the actual function
    cam_obs = []
    for cam_info in cam_infos:
        K = cam_info['K']
        TWC = Transform(cam_info['TWC'])
        resolution = cam_info['resolution']
        # Paramétrisation de la caméra
        # Utilise des fonctions de Bullet --> Semble permettre de calculer la "vue"
        # de la caméra à partir de paramètres tel que Yaw, Pitch, Roll ...
        # --> Paramétrisation standard ?
        cam = Camera(resolution=resolution, client_id=renderer.client_id)
        cam.set_intrinsic_K(K)
        cam.set_extrinsic_T(TWC)
        cam_obs_ = cam.get_state()
        if renderer.background_color is not None:
            im = cam_obs_['rgb']
            mask = cam_obs_['mask']
            im[np.logical_or(mask < 0, mask == 255)] = renderer.background_color
            if render_depth:
                depth = cam_obs_['depth']
                near, far = cam_obs_['near'], cam_obs_['far']
                z_n = 2 * depth - 1
                z_e = 2 * near * far / (far + near - z_n * (far - near))
                z_e[np.logical_or(mask < 0, mask == 255)] = 0.
                cam_obs_['depth'] = z_e
        cam_obs.append(cam_obs_)


    # cam_obs est la scène retournée ...
    # C'est le rgb render qu'on affiche ??
        
    # rgb_rendered = renderer.render_scene(list_objects, [camera])[0]['rgb']
