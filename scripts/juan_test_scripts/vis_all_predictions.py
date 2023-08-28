import gdrn_simple #Needed to append core to PYTHONPATH

import mmcv
from lib.utils.mask_utils import  get_edge
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
from core.utils.data_utils import read_image_mmcv
from core.utils.data_utils import read_image_mmcv
import gdrn_simple.DataLoadingUtils as data_utils
from gdrn_simple.DatasetConfig import get_dataset_config
from scripts.gdrn_simple.RenderClients import MyCppRenderer
import gdrn_simple.VisUtils as vis_utils

def format_pred_info(preds_csv, id2obj):
    preds = defaultdict(list) 
    for item in preds_csv:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        item["time"] = float(item["time"])
        item["score"] = float(item["score"])
        item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
        item["t"] = parse_Rt_in_csv(item["t"]) / 1000
        item["obj_name"] = id2obj[item["obj_id"]]
        preds[im_key].append(item)

    return dict(preds)

def load_predicted_csv(fname, id2obj)->Dict[str, List[Dict]]:
    """Return dict of predictions
    
    {scene_im_id: [{pred1}, {pred2}, ...], ...}

    """

    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    info_dict = format_pred_info(info_list, id2obj)

    return info_dict

def sort_dataset_by_scene_im_id(dataset)->Dict[str, List[Dict]]:
    gt_dict = {} 
    for sample in dataset:
        gt_dict[sample["scene_im_id"]] = sample
    return gt_dict

def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])

pred_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_env1_automated1/inference_model_final/ambf_suturing_test/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-ambf-suturing-test-iter0_ambf_suturing-test.csv" 

def main():
    global pred_path 
    pred_path = Path(pred_path)

    dataset_cfg = get_dataset_config("ambf_suturing")
    dataset = data_utils.SimpleDataDictDataset("ambf_suturing_test")

    print(dataset.get_stored_keys())
    print(dataset.get_annotation_keys())
    print(f"Dataset len: {len(dataset)}\n")
    print(dataset[-1]["scene_im_id"])

    preds_dict = load_predicted_csv(pred_path, dataset_cfg.ID2OBJ)
    assert len(preds_dict) == len(dataset), "Number of predictions and dataset samples do not match"

    gt_dict = sort_dataset_by_scene_im_id(dataset)

    width = 640
    height = 480
    renderer = MyCppRenderer("cpp", dataset_cfg.MODEL_PATHS, dataset_cfg.OBJID, width=width, height=height)

    for scene_im_id, pred_info in preds_dict.items():
        pred_pose = np.hstack((pred_info[0]["R"],pred_info[0]["t"].reshape(3,1)))

        gt_info = gt_dict[scene_im_id]
        gt_pose = gt_info["annotations"][0]["pose"] 
        obj_id = gt_info["annotations"][0]["category_id"]
        K = gt_info["cam"]

        rgb_im = read_image_mmcv(gt_info["file_name"])

        # Render the gt
        gt_rgb, gt_depth = renderer.render(obj_id, K, gt_pose)
        # Render the prediction 
        pred_rgb, pred_depth = renderer.render(obj_id, K, pred_pose)

        vis = vis_utils.vis_gt_and_pred(rgb_im, gt_rgb, pred_rgb)
        # vis = vis_utils.vis_gt_and_pred_option2(rgb_im, gt_rgb, pred_rgb)

        cv2.imshow(f"rgb", vis)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()