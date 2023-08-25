bop_renderer_path = '/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan/bop_renderer/'
core_path =  '/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan'
import sys
import numpy as np
import imageio
sys.path.append(bop_renderer_path)
sys.path.append(core_path)
import bop_renderer
from lib.pysixd import transform
import cv2

# PARAMETERS.
################################################################################
# Path to a 3D object model (in PLY format).
model_path ='/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan/datasets/BOP_DATASETS/ambf_suturing/models/obj_000001.ply'
obj_id = 1

# Path to output RGB and depth images.
out_rgb_path = 'out_rgb.png'
out_depth_path = 'out_depth.png'

# Object pose and camera parameters.
R = transform.random_rotation_matrix()[:3, :3].flatten().tolist()
t = [0.0, 0.0, 50.0]
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

cv2.imshow("rgb", rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the rendered images.
# imageio.imwrite(out_rgb_path, rgb)
# imageio.imwrite(out_depth_path, depth)