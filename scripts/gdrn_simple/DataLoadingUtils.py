
import gdrn_simple
from pathlib import Path
import sys
import mmcv
import numpy as np
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import pandas as pd


from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.vis_utils.image import vis_image_bboxes_cv2
from dataclasses import dataclass

@dataclass
class SimpleDataDictDataset:
    dataset_name: str

    def __post_init__(self):
        register_datasets([self.dataset_name])
        self.dataset_dicts = DatasetCatalog.get(self.dataset_name)
        self.metadata = MetadataCatalog.get(self.dataset_name) 
    
    def __len__(self):
        return len(self.dataset_dicts)
    
    def __getitem__(self, idx):
        return self.dataset_dicts[idx]
    
    def get_stored_keys(self):
        return self.dataset_dicts[0].keys()
    
    def get_annotation_keys(self):
        return self.dataset_dicts[0]["annotations"][0].keys()
    
    def get_rgb(self, idx)->np.ndarray:
        return read_image_mmcv(self.dataset_dicts[idx]["file_name"])
    
    def get_pose(self, idx)->np.ndarray:
        return self.dataset_dicts[idx]["annotations"][0]["pose"]
    
    def get_intrinsic(self, idx)->np.ndarray:
        return self.dataset_dicts[idx]["cam"]

if __name__ == "__main__":
    dataset_name = "ambf_suturing"
    dataset = SimpleDataDictDataset(dataset_name)

    print(f"Dataset len: {len(dataset)}\n")

    print("Stored keys")
    print(dataset.get_stored_keys(),"\n")

    print("Annotation's keys")
    print(dataset.get_annotation_keys(),"\n")