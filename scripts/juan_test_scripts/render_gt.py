import os.path as osp
from gdrn_simple.DataLoadingUtils import SimpleDataDictDataset
from gdrn_simple.RenderClients import RendererClient
from lib.pysixd.config import datasets_path
import os
import cv2

# name = "ambf_suturing"
# id2obj = {1: "needle"}
# obj_ids = list(id2obj.keys())
# model_paths = [os.path.join(datasets_path,"ambf_suturing/models/obj_000001.ply")]

name = "tudl_bop_test"
id2obj = {1: "dragon", 2: "frog", 3: "can"}
model_paths = [osp.join(datasets_path,"tudl/models",f"obj_{obj_id:06d}.ply") for obj_id in id2obj]
obj_ids = list(id2obj.keys())

def main():
    dataset = SimpleDataDictDataset(name) 
    print(dataset.get_stored_keys())
    print(dataset[0]["annotations"][0].keys())
    print(len(dataset[0]["annotations"]))

    for m in model_paths:
        print(m)
        assert osp.exists(m)

    rgb_im = dataset.get_rgb(0)
    pose = dataset.get_pose(0)
    K = dataset.get_intrinsic(0)

    print(rgb_im.shape)
    print(pose)
    print(K)
    print(f"obj_id {dataset[0]['annotations'][0]['category_id']}")

    renderer = RendererClient("cpp", model_paths, obj_ids, width=640, height=480)

    uniform_color = [0.0, 0.5, 0.0]
    obj_id = dataset[0]["annotations"][0]["category_id"] + 1
    ren_rgb, ren_depth = renderer.render(obj_id, K, pose, uniform_color)

    blended = cv2.addWeighted(rgb_im, 0.3, ren_rgb, 0.7, 0)
    cv2.imshow("rgb", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()