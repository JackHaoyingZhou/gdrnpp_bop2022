import gdrn_simple

from pathlib import Path
from mmcv import Config
from detectron2.data import get_detection_dataset_dicts
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList
from core.gdrn_modeling.datasets.data_loader_online import GDRN_Online_DatasetFromList 
import torch.utils.data as torchdata
from core.utils.dataset_utils import trivial_batch_collator
from core.gdrn_modeling.engine.engine_utils import batch_data,get_renderer
import ref.ambf_suturing as ambf_ds_ref
import cv2

if __name__ == '__main__':


    config_name = 'convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ambf_suturing'
    config_path = Path(f'configs/gdrn/ambf_suturing/{config_name}.py')

    cfg = Config.fromfile(config_path)
    # cfg = setup(cfg)
    register_datasets_in_cfg(cfg)

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
    dataset = GDRN_Online_DatasetFromList(cfg, split="train", lst=dataset_dicts, flatten=True)

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

    # s2 = next(iter(data_loader))
    for s2 in data_loader:
        # Process batch for model 
        renderer = get_renderer(cfg, ambf_ds_ref, obj_names=["needle"]) 
        batch = batch_data(cfg, s2, phase="train", device="cuda", renderer=renderer)

        cv2.imshow("img", batch["roi_img"][0].detach().cpu().numpy().transpose(1,2,0))
        
        q = cv2.waitKey(0)
        if q == ord("q"):
            break

    cv2.destroyAllWindows()

    x=0