import os.path as osp
from gdrn_simple.DataLoadingUtils import SimpleDataDictDataset
from gdrn_simple.RenderClients import RendererClient, MyEGLRenderer,MyCppRenderer
from gdrn_simple.DatasetConfig import get_dataset_config
import cv2

width = 640
height = 480
name = "ambf_suturing_test"
# name = "tudl_bop_test"

def main():
    dataset_cfg = get_dataset_config(name)
    dataset = SimpleDataDictDataset(dataset_cfg.NAME) 
    print(dataset.get_stored_keys())
    print(dataset[0]["annotations"][0].keys())
    print(len(dataset[0]["annotations"]))
    print(f"Dataset len: {len(dataset)}\n")

    # Two renderer options 
    renderer = MyCppRenderer("cpp", dataset_cfg.MODEL_PATHS, dataset_cfg.OBJID, width=width, height=height)
    # renderer = MyEGLRenderer("EGL", dataset_cfg.MODEL_PATHS, dataset_cfg.OBJID, width=width, height=height)

    for i in range(len(dataset)):
        rgb_im = dataset.get_rgb(i)
        pose = dataset.get_pose(i)
        K = dataset.get_intrinsic(i)

        print(rgb_im.shape)
        print(pose)
        print(K)
        print(f"obj_id {dataset[0]['annotations'][0]['category_id']}")


        obj_id = dataset[0]["annotations"][0]["category_id"] # Zero indexed
        ren_rgb, ren_depth = renderer.render(obj_id, K, pose)

        blended = cv2.addWeighted(rgb_im, 0.3, ren_rgb, 0.7, 0)
        cv2.imshow("rgb", blended)
        k = cv2.waitKey(0)

        if k == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()