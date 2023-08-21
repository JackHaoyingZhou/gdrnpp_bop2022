from dataclasses import dataclass
from typing import List
import numpy as np
from lib.pysixd.config import datasets_path
import bop_renderer

@dataclass
class RendererClient:
    renderer_type: str 
    models_path: List[str]
    obj_ids: List[int]
    width: int
    height: int
    use_uniform_color: bool = False

    def __post_init__(self):
        self.ren = self.create_render()

    def create_render(self):
        ren = bop_renderer.Renderer()
        ren.init(self.width, self.height)
        ren.set_light([0, 0, 0], [1.0, 1.0, 1.0], 0.5, 1.0, 1.0, 8.0)

        for obj_id, model in zip(self.obj_ids, self.models_path):
            ren.add_object(obj_id, model)
            
        return ren
    
    def render(self, obj_id:int, intrinsic:np.ndarray, pose:np.ndarray, uniform_color:np.ndarray):
        fx, fy, cx, cy = (
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        R = pose[:3,:3].flatten().tolist()
        t = (pose[:3,3]*1000).tolist()
        self.ren.render_object( obj_id, R,t, fx, fy, cx, cy)

        rgb = self.ren.get_color_image(obj_id)
        depth = self.ren.get_depth_image(obj_id)

        return rgb, depth