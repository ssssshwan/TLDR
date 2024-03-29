## TLDR
Official PyTorch implementation of [**"Texture Learning Domain Randomization for Domain Generalized Segmentation"**](https://arxiv.org/abs/2303.11546) (ICCV2023).

## Setup Environment

The requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html  
```

Further, please download the ImageNet pre-trained weights and a pretrained TLDR the following script.
```shell
bash scripts/download_checkpoints.sh
```

## Setup Datasets
- Download [**Cityscapes**](https://www.cityscapes-dataset.com/), [**BDDS**](https://doc.bdd100k.com/download.html), [**Mapillary**](https://www.mapillary.com/datasets) and make the directory structures as follows in **<path_to_tldr>/data** folder.

```
cityscapes
 └ leftImg8bit
     └ train
     └ val
     └ test
 └ gtFine
     └ train
     └ val
     └ test
```
```
bdd-100k
 └ images
   └ train
   └ val
   └ test
 └ labels
   └ train
   └ val
```
```
mapillary
 └ training
   └ images
   └ labels
 └ validation
   └ images
   └ labels
 └ test
   └ images
   └ labels
```

- Download [**GTA**](https://download.visinf.tu-darmstadt.de/data/from_games/) and [**SYNTHIA**](http://synthia-dataset.net/download/808/) and split them into training/validation/test set following the approach used in [**RobustNet**](https://github.com/shachoi/RobustNet/tree/main/split_data). Then, make the directory structures as follows in **<path_to_tldr>/data** folder.

```
GTAV
 └ images
   └ train
   └ valid
   └ test   
 └ labels
   └ train
   └ valid
   └ test   
```

```
synthia
 └ RGB
   └ train
   └ val
 └ GT
   └ COLOR
     └ train
     └ val
   └ LABELS
     └ train
     └ val
```


**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs.

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```
## Evaluate Pre-trained Model

We provide TLDR checkpoint trained on GTA dataset using ResNet-50 (already downloaded by `tools/download_checkpoints.sh`).
The checkpoint can be evalutated on the Cityscapes validation dataset using:

```shell
bash scripts/test.sh work_dirs/230203_0112_iter_40000_lr_3e-05_orig_0.5_style_0.5_regw_0.005_regr_1.0_disentw_0.005_disentr_1.0_threshold_0.1_seed_300_fd325
```

The provided checkpoint must achieve 46.97 mIoU, which is one of the checkpoints from the random seeds used in the experiment.


## Training
Download [**ImageNet validation**](https://image-net.org/challenges/LSVRC/index.php) dataset as a random style dataset, please place the dataset in **<path_to_tldr>/data** folder.

We provide an [config file](configs/TLDR/gta2cs_stylizations_warm_deeplabv3plus_resnet50.py) of the TLDR.
A training can be launched using:

```shell
python run_experiments.py --config configs/TLDR/gta2cs_stylizations_warm_deeplabv3plus_resnet50.py
```

## Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).

The most relevant files for TLDR are:

* [configs/TLDR/gta2cs_stylizations_warm_deeplabv3plus_resnet50.py](configs/TLDR/gta2cs_stylizations_warm_deeplabv3plus_resnet50.py):
  Annotated config file for the final TLDR.
* [mmseg/models/uda/stylization.py](mmseg/models/uda/stylization.py):
  Implementation of TLDR including texture regularization loss and texture generalization loss.
* [mmseg/datasets/photo_wct_batch.py](mmseg/datasets/photo_wct_batch.py):
  Implementation of Style Transfer Module (STM).

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [RobustNet](https://github.com/shachoi/RobustNet)
* [FastPhotoStyle](https://github.com/NVIDIA/FastPhotoStyle)
