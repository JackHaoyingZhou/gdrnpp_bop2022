import sys
from typing import Any, Dict, List
core_path = "/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan"
sys.path.append(core_path)

from pathlib import Path
import cv2
import mmcv
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
import pandas as pd

# cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge
from core.utils.data_utils import read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.vis_utils.image import grid_show
from lib.vis_utils.image import vis_image_bboxes_cv2

## Constants
score_thr = 0.3
colors = colormap(rgb=False, maximum=255)
# object info
id2obj = {1: "dragon", 2: "frog", 3: "can"}
objects = list(id2obj.values())
# Camera info
width = 640
height = 480

def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list


def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])

def load_predictions(pred_path:Path)->Dict[str, List[Dict]]:
    """
    Load predictions from csv file into a dict[str, List[Dict]].
    The keys for the dictionary have the following format: scene_id/im_id.
    For example, scene 1 and img 0 will have the key: 1/0.

    Each key can have multiple entries, since there is a prediction per bounding box.
    Therefore the value is a List of dictionaries
    """

    preds_csv = load_predicted_csv(pred_path)
    # pred_bboxes = mmcv.load(bbox_path)
    preds = {}
    for item in preds_csv:
        im_key = "{}/{}".format(item["scene_id"], item["im_id"])
        item["time"] = float(item["time"])
        item["score"] = float(item["score"])
        item["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
        item["t"] = parse_Rt_in_csv(item["t"]) / 1000
        item["obj_name"] = id2obj[item["obj_id"]]
        if im_key not in preds:
            preds[im_key] = []
        preds[im_key].append(item)
    return preds

def main():
    #Paths
    model_dir = "datasets/BOP_DATASETS/tudl/models/"
    model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

    pred_path = osp.join(
        "output/gdrn/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl/",
        "inference_model_final_wo_optim/tudl_bop_test/",
        "convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-tudl-test-iter0_tudl-test.csv",
    )
    vis_dir = "output/gdrn/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl/vis"

    bbox_path = "datasets/BOP_DATASETS/tudl/test/test_bboxes/yolox_x_640_tudl_pbr_tudl_bop_test.json"

    pred_path = Path(pred_path)
    assert pred_path.exists(), "pred path not found"

    #### Renderer
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    ren = EGLRenderer(
        model_paths,
        vertex_scale=0.001,
        use_cache=True,
        width=width,
        height=height,
    )

    #### Load dataset
    dataset_name = "tudl_bop_test"
    print(dataset_name)
    register_datasets([dataset_name])

    meta = MetadataCatalog.get(dataset_name)
    print("MetadataCatalog: ", meta)
    objs = meta.objs

    dset_dicts = DatasetCatalog.get(dataset_name)

    #### load predictions
    preds = load_predictions(pred_path)

    dset_dicts = DatasetCatalog.get(dataset_name)
    for d in tqdm(dset_dicts):
        #### Load GT data 
        K = d["cam"]
        file_name = d["file_name"]
        scene_im_id = d["scene_im_id"]  #  e.g. 1/0 for scene 1 and img 0
        print(file_name)
        img = read_image_mmcv(file_name, format="BGR")

        scene_im_id_split = d["scene_im_id"].split("/")
        scene_id = scene_im_id_split[0]
        im_id = int(scene_im_id_split[1])
        imH, imW = img.shape[:2]

        if scene_im_id not in preds:
            print(scene_im_id, "not detected")
            continue
        
        #### Extract all predictions for a given key
        cur_preds = preds[scene_im_id]
        kpts_2d_est = []
        est_Rs = []
        est_ts = []
        est_labels = []

        # print(f"current_preds: {cur_preds}")

        for pred_i, pred in enumerate(cur_preds):
            try:
                R_est = pred["R"]
                t_est = pred["t"]
                score = pred["score"]
                obj_name = pred["obj_name"]
            except:
                continue
            if score < score_thr:
                continue

            est_Rs.append(R_est)
            est_ts.append(t_est)
            est_labels.append(objects.index(obj_name))  # 0-based label
        

        #### Render all the predictions

        im_gray = mmcv.bgr2gray(img, keepdim=True)
        im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

        gt_Rs = []
        gt_ts = []
        gt_labels = []

        # 0-based label -- Load ground truth annotations
        annos = d["annotations"] #d is the sample dictionary
        cat_ids = [anno["category_id"] for anno in annos]
        obj_names = [objs[cat_id] for cat_id in cat_ids]

        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        for anno_i, anno in enumerate(annos):
            obj_name = obj_names[anno_i]
            gt_labels.append(objects.index(obj_name))  # 0-based label

            gt_Rs.append(Rs[anno_i])
            gt_ts.append(transes[anno_i])

        est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]
        gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]

        ren.render(
            est_labels,
            est_poses,
            K=K,
            image_tensor=image_tensor,
            background=im_gray_3,
        )
        ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

        for gt_label, gt_pose in zip(gt_labels, gt_poses):
            ren.render([gt_label], [gt_pose], K=K, seg_tensor=seg_tensor)
            gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
            gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
            ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))

        for est_label, est_pose in zip(est_labels, est_poses):
            print(est_pose)
            ren.render([est_label], [est_pose], K=K, seg_tensor=seg_tensor)
            est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
            est_edge = get_edge(est_mask, bw=3, out_channel=1)
            ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

        vis_im = ren_bgr

        show = True 
        if show:
            # im_show = cv2.hconcat([img, vis_im, vis_im_add])
            im_show = cv2.hconcat([img, vis_im])
            cv2.imshow("im_est", im_show)
            if cv2.waitKey(0) == 27:
                break  # esc to quit

        break

if __name__ == "__main__":
    main()


