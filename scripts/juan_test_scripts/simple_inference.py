import sys

core_path = "/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan"
sys.path.append(core_path)

from torch.cuda.amp import autocast
import numpy as np
import os.path as osp
from core.utils.data_utils import read_image_mmcv
from pathlib import Path
import mmcv
from mmcv import Config
from core.gdrn_modeling.models import (
    GDRN,
    GDRN_no_region,
    GDRN_cls,
    GDRN_cls2reg,
    GDRN_double_mask,
    GDRN_Dstream_double_mask,
)  # noqa
from core.utils.my_checkpoint import MyCheckpointer
from detectron2.data import get_detection_dataset_dicts
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList
import torch.utils.data as torchdata
from core.utils.dataset_utils import trivial_batch_collator
from core.gdrn_modeling.engine.engine_utils import batch_data
import torch
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
import cv2
from lib.utils.mask_utils import mask2bbox_xyxy, cocosegm2mask, get_edge

def setup(cfg:Config)->Config:
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        print("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
        # if accumulate_iter > 1:
        #     if "weight_decay" in cfg.SOLVER.OPTIMIZER_CFG:
        #         cfg.SOLVER.OPTIMIZER_CFG["weight_decay"] *= (
        #             cfg.SOLVER.IMS_PER_BATCH * accumulate_iter / bs_ref
        #         )  # scale weight_decay

    return cfg

def load_dataset(cfg):

    dataset_name = cfg.DATASETS.TEST[0]
    print(dataset_name)

    # Load dataset
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    cfg.TEST.TEST_BBOX_TYPE="gt" #needed to avoir failure in line 754 of indata_loader.py  
    dataset = GDRN_DatasetFromList(cfg, split="test", lst=dataset_dicts, flatten=False)

    s1 = dataset[0]  # images are still not loaded in this point.

    flag = isinstance(dataset, torchdata.IterableDataset)
    print(f"Is torchdata.IterableDataset: {flag}")

    # DataLoader
    data_loader = torchdata.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=trivial_batch_collator,
        pin_memory=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        persistent_workers=cfg.DATALOADER.PERSISTENT_WORKERS,
    )

    return data_loader

if __name__ == '__main__':
    config_name = 'convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl'
    config_path = Path(f'configs/gdrn/tudl/{config_name}.py')
    weights_path = f'./output/pretrained/tudl/{config_name}/model_final_wo_optim.pth'
    weights_path = Path(weights_path)

    cfg = Config.fromfile(config_path)
    cfg = setup(cfg)
    register_datasets_in_cfg(cfg)

    print(cfg.MODEL.POSE_NET.NAME)

    model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=True)
    MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    data_loader = load_dataset(cfg)

    # Sample bath
    iterator = iter(data_loader)
    s2 = next(iterator)
    s2 = next(iterator)
    batch = batch_data(cfg, s2, phase="test", device=cfg.MODEL.DEVICE)

    
    if cfg.INPUT.WITH_DEPTH and "depth" in cfg.MODEL.POSE_NET.NAME.lower():
        print("using depth ...")
        inp = torch.cat([batch["roi_img"], batch["roi_depth"]], dim=1)
    else:
        inp = batch["roi_img"]

    print("inference")
    with inference_context(model), torch.no_grad():
        amp_test = False
        with autocast(enabled=amp_test):
            out_dict = model(
                inp,
                roi_classes=batch["roi_cls"],
                roi_cams=batch["roi_cam"],
                roi_whs=batch["roi_wh"],
                roi_centers=batch["roi_center"],
                resize_ratios=batch["resize_ratio"],
                roi_coord_2d=batch.get("roi_coord_2d", None),
                roi_coord_2d_rel=batch.get("roi_coord_2d_rel", None),
                roi_extents=batch.get("roi_extent", None),
            )

    #### Analysis
    pred_trans = out_dict['trans'][0].detach().cpu().numpy()
    pred_rot = out_dict['rot'][0].detach().cpu().numpy()

    gt_trans = s2[0]['annotations'][0]['trans']
    gt_rot = quat2mat(s2[0]['annotations'][0]['quat'])

    scene_img_id = f"{s2[0]['scene_im_id']}"
    print(f"scene_img_id {scene_img_id}")
    print(f"caterogy id {s2[0]['annotations'][0]['category_id']}" )
    print(f"results for {s2[0]['file_name']}") 

    print(f"predictions")
    print(f"trans {pred_trans}")
    print(f"rot {pred_rot}")

    print(f"ground-truth")
    print(f"trans {gt_trans}")
    print(f"rot {gt_rot}")

    print(f"diff")
    print(f"trans diff {gt_trans-pred_trans}")
    print(f"rot diff {gt_rot-pred_rot}")

    ## Create renderer
    id2obj = {1: "dragon", 2: "frog", 3: "can"}
    objects = list(id2obj.values())
    # Camera info
    width = 640
    height = 480
    model_dir = "datasets/BOP_DATASETS/tudl/models/"
    model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

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

    ##Render
    s2 = s2[0]
    K = s2["cam"][0]
    file_name = s2["file_name"][0]
    scene_im_id = s2["scene_im_id"][0] #  e.g. 1/0 for scene 1 and img 0
    print(file_name)
    img = read_image_mmcv(file_name, format="BGR")
    est_pose = np.hstack([pred_rot, pred_trans.reshape(3, 1)])
    est_label = 0

    im_gray = mmcv.bgr2gray(img, keepdim=True)
    im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

    ren.render(
        est_label,
        est_pose,
        K=K,
        image_tensor=image_tensor,
        background=im_gray_3,
    )
    ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

    ##Actual rendering of predictions
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
        cv2.waitKey(0)
        cv2.destroyAllWindows()