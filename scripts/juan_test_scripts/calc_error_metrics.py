
from typing import Any, Dict, List
import gdrn_simple #Needed to append core to PYTHONPATH
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from core.utils.data_utils import read_image_mmcv
from core.utils.data_utils import read_image_mmcv
import gdrn_simple.DataLoadingUtils as data_utils
from gdrn_simple.DatasetConfig import get_dataset_config
from scripts.gdrn_simple.RenderClients import MyCppRenderer
import gdrn_simple.VisUtils as vis_utils
from gdrn_simple.Metrics import PredictionRecord as PR
from gdrn_simple.Metrics import ErrorRecordHeader as ERH 
from gdrn_simple.Metrics import ErrorRecord 
from lib.pysixd.pose_error import te, re, mssd
from lib.pysixd.inout import load_ply

class ErrorMetricsCalculator:
    def __init__(self, models_pts:Dict[int, np.ndarray]):
        self.models_pts = models_pts
        self.data = {ERH.scene_im_id:[],
                ERH.scene_id:[],
                ERH.re:[],
                ERH.te:[],
                ERH.mssd:[]}

        self.model_sym = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}] 

    def calculate_error_metrics(self, scene_im_id:str, scene_id:str, gt_pose, est_pose): 
        gt_R = gt_pose[:3,:3] 
        gt_t = gt_pose[:3,3].reshape((3,1))
        est_R = est_pose[:3,:3]
        est_t = est_pose[:3,3].reshape((3,1))

        
        re_error = re(est_R, gt_R) 
        te_error = te(est_t, gt_t)* 1000
        mssd_error = mssd(est_R, est_t,gt_R, gt_t, self.models_pts[1]['pts'], self.model_sym)

        self.data[ERH.scene_im_id].append(scene_im_id) 
        self.data[ERH.scene_id].append(scene_id)
        self.data[ERH.re].append(re_error)
        self.data[ERH.te].append(te_error)
        self.data[ERH.mssd].append(mssd_error)

    def get_error_metrics(self)->Dict[ERH, List[Any]]:
        return self.data 

# pred_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_env1_automated1/inference_model_final/ambf_suturing_test/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-ambf-suturing-test-iter0_ambf_suturing-test.csv" 
pred_path = "./output/gdrn/ambf_suturing/classAware_ambf_suturing_v0.0.1/inference_model_final/ambf_suturing_test/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-ambf-suturing-test-iter0_ambf_suturing-test.csv" 

def main():
    global pred_path 
    pred_path = Path(pred_path)

    dataset_cfg = get_dataset_config("ambf_suturing")
    dataset = data_utils.SimpleDataDictDataset("ambf_suturing_test")
    gt_dict = PR.sort_dataset_by_scene_im_id(dataset)

    preds_dict = PR.load_predicted_csv(pred_path, dataset_cfg.ID2OBJ)
    assert len(preds_dict) == len(dataset), "Number of predictions and dataset samples do not match"

    # Load model points for mssd metric
    models = {}
    for obj_id in dataset_cfg.OBJID:
        models[obj_id] = load_ply(dataset_cfg.MODEL_PATHS[obj_id-1])

    error_metrics_calculator = ErrorMetricsCalculator(models)

    for scene_im_id, pred_info in preds_dict.items():
        pred_pose = np.hstack((pred_info[0]["R"],pred_info[0]["t"].reshape(3,1)))

        gt_info = gt_dict[scene_im_id]
        gt_pose = gt_info["annotations"][0]["pose"] 
        scene_im_id = gt_info["scene_im_id"]
        scene_id = scene_im_id.split("/")[0]
        obj_id = gt_info["annotations"][0]["category_id"]
        img_id = gt_info["image_id"]
        K = gt_info["cam"]

        error_metrics_calculator.calculate_error_metrics(scene_im_id, scene_id, gt_pose, pred_pose)

    error_record = ErrorRecord() 
    error_record.add_data(error_metrics_calculator.get_error_metrics())
    df = error_record.generate_df()

    print("\n\n")
    # Conver to csv online with https://tableconvert.com/markdown-to-csv
    for metric in [ERH.re, ERH.te, ERH.mssd]:
        table = error_record.generate_summary_table(metric)
        print(f"## {metric.value} table")
        table.print(floatfmt="0.2f")

if __name__ == "__main__":
    main()