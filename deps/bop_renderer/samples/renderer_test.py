
"""Comparison of the BOP Renderer and the Python renderer from BOP Toolkit."""

import sys
import time
import numpy as np


# PARAMETERS.
################################################################################
# Path to bop_renderer.
bop_renderer_path = '/path/to/bop_renderer/build'

# Path to bop_toolkit which contains the Python renderer.
bop_toolkit_path = '/path/to/bop_toolkit'

# Path to a 3D object model (in PLY format).
model_path = '/datasets/bop/lm/models/obj_000001.ply'

# Object pose and camera parameters.
R = np.eye(3)
t = np.array([[0.0, 0.0, 150.0]]).T
fx, fy, cx, cy = 572.41140, 573.57043, 325.26110, 242.04899
im_size = (640, 480)
################################################################################


# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
sys.path.append(bop_renderer_path)
import bop_renderer

sys.path.append(bop_toolkit_path)
from bop_toolkit_lib import inout, renderer_py

# Init the renderers.
# ------------------------------------------------------------------------------
# Init the C++ renderer.
ren = bop_renderer.Renderer()
ren.init(im_size[0], im_size[1])
obj_id = 1
ren.add_object(obj_id, model_path)

# Init the Python renderer.
ren_py = renderer_py.RendererPython(im_size[0], im_size[1])
obj_id = 1
ren_py.add_object(obj_id, model_path)

# Object model and camera matrix (will be used by the Python renderer).
model = inout.load_ply(model_path)
K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
K = np.array(K).reshape((3, 3))

# Measure rendering time.
# ------------------------------------------------------------------------------
renderer_types = ['cpp', 'py']
for renderer_type in renderer_types:
  times = []
  for i in range(100):
    if i % 10 == 0:
      print(i)

    t_start = time.time()

    if renderer_type == 'cpp':
      R_list = R.flatten().tolist()
      t_list = t.flatten().tolist()
      ren.render_object(obj_id, R_list, t_list, fx, fy, cx, cy)
      rgb = ren.get_color_image(obj_id)
      depth = ren.get_depth_image(obj_id)
    else:
      res_dict = ren_py.render_object(obj_id, R, t, fx, fy, cx, cy)
      depth_p, rgb_p = res_dict['depth'], res_dict['rgb']
    times.append(time.time() - t_start)

  print('Average rendering time for {} renderer: {}'.format(
    renderer_type, np.mean(times)))

# Compare results of the C++ and the Python renderer.
# ------------------------------------------------------------------------------
# C++ renderer.
R_list = R.flatten().tolist()
t_list = t.flatten().tolist()
ren.render_object(obj_id, R_list, t_list, fx, fy, cx, cy)
rgb_c = ren.get_color_image(obj_id)
depth_c = ren.get_depth_image(obj_id)

# Python renderer.
res_dict = ren_py.render_object(obj_id, R, t, fx, fy, cx, cy)
depth_p, rgb_p = res_dict['depth'], res_dict['rgb']

# Difference of the RGB renderings.
rgb_diff = np.abs(
  rgb_c.astype(np.float) - rgb_p.astype(np.float)).astype(np.uint8)

# Difference of the depth renderings.
depth_diff = np.abs(
  depth_c.astype(np.float) - depth_p.astype(np.float)).astype(np.uint16)

# Print statistics of the differences.
print('rgb diff sum: {}'.format(rgb_diff.sum()))
print('rgb diff mean: {}'.format(rgb_diff.mean()))
print('rgb non-zero diff count: {}'.format((rgb_diff > 0).sum()))
print('rgb non-zero diff mean: {}'.format(rgb_diff[rgb_diff > 0].mean()))
print('depth diff sum: {}'.format(depth_diff.sum()))
print('depth diff mean: {}'.format(depth_diff.mean()))
