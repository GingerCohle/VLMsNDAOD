# VLM-empowered Nighttime DAOD with Deformable Part-based Sampling and SNR-based Graph Matching



## Installation

#### Our work is based on Python 3.7 and Pytorch 1.9.1+cu111 due to the  [CLIP requirement](https://github.com/openai/CLIP). The hardware is Nvidia RTX 4090. Give a big thanks to [SIGMA](https://github.com/CityU-AIM-Group/SIGMA). We use it as baseline.

##### Step 1:

##### Please git clone ths repo and cd the repo dir.

##### Step 2:

```bash
conda create -n vldadaptor  python==3.7 -y
conda activate vldadaptor
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
conda install ipython
pip install ninja yacs cython matplotlib tqdm 
pip install --no-deps torchvision==0.2.1 
python setup.py build_ext install
cd ../..
pip install opencv-python==3.4.17.63
pip install scikit-learn
pip install scikit-image
python setup.py build develop
pip install Pillow==7.1.0
pip install tensorflow tensorboardX
pip install ipdb
```

#### CLIP Installation (China Region)

```bash
pip install ftfy regex tqdm
pip install git+https://gitee.com/lazybeauty/CLIP.git
```

#### CLIP Installation (Other Regions)

```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Dataset

### Give a big thank to the author of [2PCNet](https://github.com/mecarill/2pcnet). The Shift dataset and BDD 100k daynight dataset are deployed as the below link. All datasets are in MSCOCO format.

### The [FLIR](https://www.flir.com/oem/adas/adas-dataset-form/) dataset is classified into daytime and nighttime dataset under very strict rule by us.

## Training

#### Foggy Cityscapes Training 

```bash
python tools/train_net_da.py --config-file configs/SIGMA/res50_nightshift.yaml
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --config-file configs/SIGMA/res50_nightshift.yaml MODEL.WEIGHT $model path$
```

