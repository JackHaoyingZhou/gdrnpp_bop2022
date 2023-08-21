import os.path as osp
from gdrn_simple.DataLoadingUtils import SimpleDataDictDataset
from gdrn_simple.RenderClients import RendererClient
from gdrn_simple.DatasetConfig import get_dataset_config
from lib.pysixd.config import datasets_path
import os
import cv2

name = "ambf_suturing"
# name = "tudl_bop_test"

def main():
    dataset_cfg = get_dataset_config(name)
    dataset = SimpleDataDictDataset(dataset_cfg.NAME) 
    print(dataset.get_stored_keys())
    print(dataset[0]["annotations"][0].keys())
    print(len(dataset[0]["annotations"]))

    rgb_im = dataset.get_rgb(0)
    pose = dataset.get_pose(0)
    K = dataset.get_intrinsic(0)

    print(rgb_im.shape)
    print(pose)
    print(K)
    print(f"obj_id {dataset[0]['annotations'][0]['category_id']}")

    renderer = RendererClient("cpp", dataset_cfg.MODEL_PATHS, dataset_cfg.OBJID, width=640, height=480)

    uniform_color = [0.0, 0.5, 0.0]
    obj_id = dataset[0]["annotations"][0]["category_id"] + 1
    ren_rgb, ren_depth = renderer.render(obj_id, K, pose, uniform_color)

    blended = cv2.addWeighted(rgb_im, 0.3, ren_rgb, 0.7, 0)
    cv2.imshow("rgb", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()