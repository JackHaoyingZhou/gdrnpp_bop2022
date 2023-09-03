# Installation instructions

### ENVIRONMENT
OS: Ubuntu 20.04
GPU: RTX 3090

## Directories structure
Save datasets using the following structure. All BOP datasets should be inside BOP_DATASETS folder.

```
gdrn_root/
└── datasets/
    ├── BOP_DATASETS/
    │   ├── ambf_suturing
    │   └── tudl
    └── VOCdevkit/
        └── VOC2012
```

VOCdevkit can be downloaded from this [webpage](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) or using the following command:

```bash
wget http://pjreddie.com/media/files/VOC2012test.tar
```

## Instructions
Instructions to install python dependencies. Note that the environment has many dependencies and therefore, it might not not be easy to install.. Detectron is precompiled against specific versions of pytorch and cuda. If a new version of pytorch is needed detectron also needs to be updated.

``` bash
#1. create conda env
# torch==1.10.1+cu113 works detectron v0.6 
conda create -n gdrnpp python=3.9 -y 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

#2. Install pytorch3d (don't forget its dependencies).
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

#3. Install detectron for torch 1.10.1+cu113
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

#3.5 Test detectron (Optional)
python scripts/detectron_simple_test/detectron_test.py

#4. Run requirements.txt - requirements.txt changed pytorch version on the process.
sh scripts/install_deps.sh no

```

### Notes:
* step 4 broke opencv installation. You can fix it with `pip uninstall opencv-python` followed by `pip install opencv-python`.

## Additional dependencies 
1. [Compile bop_renderer](#compiling-bop_renderer). 

2. Modify [config.py](./lib/pysixd/config.py) in pysixd folder.
    * Change `BOP_RENDERER_PATH` to the path of the compiled bop_renderer.
    * Change `BOP_PATH` to the path of the BOP dataset.

3. Soft link bob_renderer to root 

```
ln -s /home/jbarrag3/research_juan/gdr-net-6dpose/gdrnpp_bop2022/bop_renderer ./bop_renderer
```
4. Compile egl_renderer
```bash
cd ./lib/egl_renderer
sh compile_cpp_egl_renderer.sh
```


### Compiling BOP_renderer

Download repo: https://github.com/thodan/bop_renderer

Make sure to activate the correct python environment before compiling the BOP_renderer

```
sudo apt install libosmesa6-dev
sudo apt-get install python-dev
git clone git@github.com:thodan/bop_renderer.git
conda activate <ENV_NAME>
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python)
cmake --build build
```

## Testing installation with GDRN simple scripts

The following scripts should run after setting up the dataset and the anaconda environment. All commands below assume you are located in the root folder of the gdrnpp repo. Make sure to install `gdrnsimple` with 

```bash
pip install -e ./scripts
```

### Testing scripts

**Testing detectron**
```bash
python3 scripts/detectron_simple_test/detectron_test.py
```

**test data loader**
```bash
python3 scripts/juan_test_scripts/load_dataset.py
```

**test data loader and renderer**
```bash
python3 scripts/juan_test_scripts/render_gt.py
```

## Original training and testing scripts for gdrn and yolox

1. For training, generated fps points for each dataset. See for example [tudl_1_comput_fps.py](./core/gdrn_modeling/tools/tudl/tudl_1_compute_fps.py)

2. Download pretrained models and use grdn and yolox test scripts.

Yolox test
```bash
./det/yolox/tools/test_yolox.sh  ./configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_tudl_real_pbr_tudl_bop_test.py 0 ./output/pretrained/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_tudl_real_pbr_tudl_bop_test/model_final.pth
```

gdrn test
```bash
./core/gdrn_modeling/test_gdrn.sh configs/gdrn/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl.py 0 ./output/pretrained/tudl/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tudl/model_final_wo_optim.pth
```
