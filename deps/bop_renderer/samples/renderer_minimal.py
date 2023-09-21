
"""A minimal example on how to use the Python bindings of the BOP Renderer."""

import numpy as np
import imageio

from bop_toolkit_lib import transform
import bop_renderer


# PARAMETERS.
################################################################################
# Path to a 3D object model (in PLY format).
model_path = '/path/to/a/ply/file'
obj_id = 1

# Path to output RGB and depth images.
out_rgb_path = 'out_rgb.png'
out_depth_path = 'out_depth.png'

# Object pose and camera parameters.
R = transform.random_rotation_matrix()[:3, :3].flatten().tolist()
t = [0.0, 0.0, 300.0]
fx, fy, cx, cy = 572.41140, 573.57043, 325.26110, 242.04899
im_size = (640, 480)
use_uniform_color = False
uniform_color = [0.0, 0.5, 0.0]
################################################################################


# Initialization of the renderer.
ren = bop_renderer.Renderer()
ren.init(im_size[0], im_size[1])
ren.set_light([0, 0, 0], [1.0, 1.0, 1.0], 0.5, 1.0, 1.0, 8.0)
ren.add_object(obj_id, model_path)

# Rendering.
ren.render_object(
  obj_id, R, t, fx, fy, cx, cy,
  use_uniform_color=use_uniform_color,
  uniform_color_r=uniform_color[0],
  uniform_color_g=uniform_color[1],
  uniform_color_b=uniform_color[2])
rgb = ren.get_color_image(obj_id)
depth = ren.get_depth_image(obj_id)

# Save the rendered images.
imageio.imwrite(out_rgb_path, rgb)
imageio.imwrite(out_depth_path, depth)
