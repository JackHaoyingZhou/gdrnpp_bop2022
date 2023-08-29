from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from lib.pysixd.config import datasets_path
import bop_renderer
import os.path as osp
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from abc import ABC, abstractmethod



@dataclass
class RendererClient(ABC):
    renderer_type: str 
    models_path: List[str]
    obj_ids: List[int]
    width: int
    height: int

    def __post_init__(self):
        self.verify_model_paths()
        self.ren = self.create_render()

    def verify_model_paths(self):
        for m in self.models_path:
            print(m)
            assert osp.exists(m)

    @abstractmethod
    def create_render(self):
        ren = bop_renderer.Renderer()
        ren.init(self.width, self.height)
        ren.set_light([0, 0, 0], [1.0, 1.0, 1.0], 0.5, 1.0, 1.0, 8.0)

        for obj_id, model in zip(self.obj_ids, self.models_path):
            ren.add_object(obj_id, model)
            
        return ren

    @abstractmethod    
    def render(self, obj_id:int, intrinsic:np.ndarray, pose:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        pass

@dataclass
class MyEGLRenderer(RendererClient):

    def create_render(self):
        ren = EGLRenderer(self.models_path,
                        vertex_scale=0.001,
                        use_cache=True,
                        width=self.width,
                        height=self.height,
                        )
        return ren

    def render(self, obj_id:int, intrinsic:np.ndarray, pose:np.ndarray):
        tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
        image_tensor = torch.empty((self.height, self.width, 4), **tensor_kwargs).detach()
        self.ren.render(
            obj_id,
            pose,
            K=intrinsic,
            image_tensor=image_tensor,
            # background=im_gray_3,
        )
        ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

        return ren_bgr, None

@dataclass
class MyCppRenderer(RendererClient):

    def create_render(self):
        ren = bop_renderer.Renderer()
        ren.init(self.width, self.height)
        ren.set_light([0, 0, 0], [1.0, 1.0, 1.0], 0.5, 1.0, 1.0, 8.0)

        for obj_id, model in zip(self.obj_ids, self.models_path):
            ren.add_object(obj_id, model)
            
        return ren
    
    def render(self, obj_id:int, intrinsic:np.ndarray, pose:np.ndarray):
        obj_id += 1 # Assume obj_id starts from 0

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