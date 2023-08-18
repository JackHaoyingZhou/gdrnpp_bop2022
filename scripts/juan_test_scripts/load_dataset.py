import sys
core_path = "/home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022_juan"
sys.path.append(core_path)
from pathlib import Path
from mmcv import Config
from detectron2.data import get_detection_dataset_dicts
from core.gdrn_modeling.datasets.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.datasets.data_loader import GDRN_DatasetFromList
import torch.utils.data as torchdata
from core.utils.dataset_utils import trivial_batch_collator
from core.gdrn_modeling.engine.engine_utils import batch_data
import cv2

if __name__ == '__main__':

    config_name = 'convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl'
    config_path = Path(f'configs/gdrn/tudl/{config_name}.py')

    cfg = Config.fromfile(config_path)
    # cfg = setup(cfg)
    register_datasets_in_cfg(cfg)

    dataset_name = cfg.DATASETS.TEST[0]
    print(dataset_name)


    '''
    Detection datasets - needs registering - Uses dectron libraries
    Returns list of dicts

    Each entry in the list is a dict with the following keys:
    (['dataset_name', 'file_name', 'depth_file', 'height', 'width', 'image_id', 
      'scene_im_id', 'cam', 'depth_factor', 'img_type', 'annotations'])

    The annotations are a dictionary with the following keys:
    
    `dataset_dicts[1]['annotations'][0].keys()`

    ['category_id', 'bbox', 'bbox_obj', 'bbox_mode', 'pose', 'quat', 'trans', 'centroid_2d',
     'segmentation', 'mask_full', 'visib_fract', 'xyz_path', 'model_info', 'bbox3d_and_center']
    '''
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

    s2 = next(iter(data_loader))

    # Process batch for model 
    batch = batch_data(cfg, s2, phase="test", device="cpu")

    cv2.imshow("img", batch["roi_img"][0].numpy().transpose(1,2,0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x=0