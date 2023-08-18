import sys
core_path = "/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan"
sys.path.append(core_path)

from pathlib import Path
from mmcv import Config
from core.utils.my_checkpoint import MyCheckpointer
from core.gdrn_modeling.models import (
    GDRN,
    GDRN_no_region,
    GDRN_cls,
    GDRN_cls2reg,
    GDRN_double_mask,
    GDRN_Dstream_double_mask,
)  # noqa


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
if __name__ == '__main__':
    config_name = 'convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl'
    weights_path = f'./output/pretrained/tudl/{config_name}/model_final_wo_optim.pth'
    weights_path = Path(weights_path)
    # weights_name ='convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl/model_final_wo_optim.pth' 
    # weights_path = Path(weights_path)/ weights_name

    config_path = Path(f'configs/gdrn/tudl/{config_name}.py')

    cfg = Config.fromfile(config_path)
    cfg = setup(cfg)
    cfg.MODEL.WEIGHTS = str(weights_path)
    print(cfg.MODEL.POSE_NET.NAME)

    model, optimizer = eval(cfg.MODEL.POSE_NET.NAME).build_model_optimizer(cfg, is_test=True)

    #Load checkpoint
    MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR, prefix_to_remove="_module.").resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    x=0