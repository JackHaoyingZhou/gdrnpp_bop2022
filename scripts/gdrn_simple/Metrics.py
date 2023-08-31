from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import gdrn_simple #Needed to append core to PYTHONPATH
import mmcv
from lib.utils.mask_utils import  get_edge
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np
import pandas as pd
from core.utils.data_utils import read_image_mmcv
from core.utils.data_utils import read_image_mmcv
import gdrn_simple.DataLoadingUtils as data_utils
from gdrn_simple.DatasetConfig import get_dataset_config
from scripts.gdrn_simple.RenderClients import MyCppRenderer
import gdrn_simple.VisUtils as vis_utils
from gdrn_simple.MkDownTable import MarkdownTable

class PredictionRecord:
    """Utility function to read csv file with network predictions"""
    @staticmethod
    def format_pred_info(preds_csv, id2obj):
        preds = defaultdict(list) 
        for item in preds_csv:
            im_key = "{}/{}".format(item["scene_id"], item["im_id"])
            item["time"] = float(item["time"])
            item["score"] = float(item["score"])
            item["R"] = PredictionRecord.parse_Rt_in_csv(item["R"]).reshape(3, 3)
            item["t"] = PredictionRecord.parse_Rt_in_csv(item["t"]) / 1000
            item["obj_name"] = id2obj[item["obj_id"]]
            preds[im_key].append(item)

        return dict(preds)

    @staticmethod
    def load_predicted_csv(fname, id2obj)->Dict[str, List[Dict]]:
        """Return dict of predictions
        {scene_im_id: [{pred1}, {pred2}, ...], ...}
        """

        df = pd.read_csv(fname)
        info_list = df.to_dict("records")
        info_dict = PredictionRecord.format_pred_info(info_list, id2obj)

        return info_dict

    @staticmethod
    def sort_dataset_by_scene_im_id(dataset)->Dict[str, List[Dict]]:
        gt_dict = {} 
        for sample in dataset:
            gt_dict[sample["scene_im_id"]] = sample
        return gt_dict

    @staticmethod
    def parse_Rt_in_csv(_item):
        return np.array([float(i) for i in _item.strip(" ").split(" ")])


class ErrorRecordHeader(Enum):
    """Headers to build a pandas dataframe with error metrics"""

    # Info
    scene_im_id = auto()
    scene_id = auto()
    im_id = auto()
    obj_id = auto()
    # Rotation and translation error
    re = "rotation error (deg)"
    te = "translation error (mm)" 
    # delta x
    dx = auto()
    dy = auto()
    dz = auto()
    # mssd - 
    mssd = "Maximum Symmetry-Aware Surface Distance (mm)"

    @classmethod
    def is_a_error_metric(cls, header):
        error_list = [cls.re, cls.te, cls.dx, cls.dy, cls.dz, cls.mssd]
        return header in error_list

class PandasOperators(Enum):
    N = auto()
    mean = auto()
    std = auto()
    max = auto()
    min = auto()
    median = auto()

    @classmethod
    def apply_op(cls, op:PandasOperators, df:pd.Series)->float:
        if op == cls.N:
            return df.count()
        elif op == cls.mean:
            return df.mean()
        elif op == cls.std:
            return df.std()
        elif op == cls.max:
            return df.max()
        elif op == cls.min:
            return df.min()
        elif op == cls.median:
            return df.median()
        

@dataclass
class ErrorRecord:

    def __post_init__(self): 
        self.ERH = ErrorRecordHeader
        self.p_op = PandasOperators

        self.data_dict = {}#defaultdict(list)
        for header in ErrorRecordHeader:
            self.data_dict[header] = []
    
    def add_data(self, data:Dict[ErrorRecordHeader, List[Any]]):
        if ErrorRecordHeader.scene_im_id not in data:
            raise ValueError("data must contain at least a scene_im_id")

        for key, list_of_vals in data.items():
            self.data_dict[key].extend(list_of_vals)

    def generate_df(self)->pd.DataFrame:
        processed_dict = self.replace_enum_for_str(self.data_dict) 
        processed_dict = self.remove_empty_cols(processed_dict)
        self.df = pd.DataFrame(processed_dict)

        return self.df
    
    def generate_summary_table(self, error_metric:ErrorRecordHeader)->MarkdownTable:
        if not ErrorRecordHeader.is_a_error_metric(error_metric):
            raise ValueError(f"{error_metric} is not a valid error metric")
        if not hasattr(self, 'df'):
            self.generate_df()

        scene_ids_list = self.df[ErrorRecordHeader.scene_id.name].unique()
        scene_ids_list.sort()
        scene_ids_list = scene_ids_list.tolist()

        # operations = ["N", "mean", "std", "min", "max"]
        operations = [self.p_op.N, self.p_op.mean, self.p_op.std, self.p_op.min, self.p_op.max]  
        table_data = []
        for op in operations:
            d1 = {"scene_id": op.name}
            for scene_id in scene_ids_list:
                sub_df = self.df.loc[self.df[self.ERH.scene_id.name]==scene_id] 
                N = self.p_op.apply_op(op, sub_df[error_metric.name])
                d1.update({scene_id: N})

            # d1.update({"total": self.df.count()[self.ERH.scene_id.name]})
            d1.update({"total": self.p_op.apply_op(op, self.df[error_metric.name])})
            table_data.append(d1)

        table = MarkdownTable(headers=["scene_id"]+scene_ids_list+["total"]) 
        for d1 in table_data:
            table.add_data(**d1)

        return table

    @staticmethod
    def replace_enum_for_str(dict): 
        new_dict = {}
        for key, val in dict.items():
            if isinstance(key, Enum):
                new_dict[key.name] = val
            else:
                new_dict[key] = val
        return new_dict

    @staticmethod
    def remove_empty_cols(dict):
        new_dict = {}
        for key, val in dict.items():
            if len(val) != 0:
                new_dict[key] = val
        return new_dict
