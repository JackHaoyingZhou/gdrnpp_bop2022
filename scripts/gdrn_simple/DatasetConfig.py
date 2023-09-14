from lib.pysixd.config import datasets_path
import os.path as osp

def get_dataset_config(name:str):
    dataset_dicts ={
        "ambf_suturing": DatasetConfig("ambf_suturing", "ambf_suturing", {1: "needle"}),
        "ambf_suturing_test": DatasetConfig("ambf_suturing_test", "ambf_suturing", {1: "needle"}),
        "tudl_bop_test": DatasetConfig("tudl_bop_test", "tudl", {1: "dragon", 2: "frog", 3: "can"}),
    } 
    return dataset_dicts[name]

class DatasetConfig:
    def __init__(self, name: str, data_root:str, id2obj: dict):
        self._NAME = name
        self._ID2OBJ = id2obj 
        self._DATA_ROOT = data_root
        self._MODEL_PATHS = [osp.join(datasets_path, self._DATA_ROOT, "models",
                                       f"obj_{obj_id:06d}.ply") for obj_id in id2obj]
        self._OBJID = list(id2obj.keys())

    @property
    def ID2OBJ(self):
        return self._ID2OBJ 
    @property
    def NAME(self):
        return self._NAME 
    @property
    def DATA_ROOT(self):
        return self._DATA_ROOT
    @property
    def MODEL_PATHS(self):
        return self._MODEL_PATHS
    @property
    def OBJID(self):
        return self._OBJID

if __name__ =="__main__":
    config = DatasetConfig("tudl_bop_test", "datasets/BOP_DATASETS/tudl", {1: "dragon", 2: "frog", 3: "can"})
    print(config.ID2OBJ)
    print(config.NAME)
    config.NAME= "hello"
    print(config.NAME)
