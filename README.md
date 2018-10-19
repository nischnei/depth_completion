# Depth Completion Selection

## Introduction

This code is based on our work [Sparsity Invariant CNNs](https://arxiv.org/pdf/1708.06500.pdf).
It is a collection of simple networks to do the task of depth completion on the [KITTI depth completion challenge](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

Feedback and code contributions are highly appreciated.

## Installation

To use this code, you have to install tensorflow and numpy. 

### tensorflow
I prefer to install the gpu version of tensorflow using pip:
```
pip install tensorflow-gpu
```

I tested this version with tensorflow 1.11.0.

### numpy
You can get numpy as well via pip
```
pip install numpy
```

I tested this version with numpy 1.14.3.


## Getting the dataset
There are some useful scripts in the 'scripts directory, check them out.

### Downloading the KITTI dataset
Just use the script 'download_all_kitti.sh' - it downloads the raw dataset as well as the depth completion data.

### generate dataset files
For training, I use textline datasets. They are generated using the 'createDataset.py' script.

! There are already dataset files in the datasets folder - if you only want to use them, skip this part !

One textline dataset includes e.g. raw velodyne projections or another e.g. the dense depth (labels).
The script expects a dataset name and a regular expression for files for which it 'globs'.

```
./createDataset.py sparse_train "/path/to/your/dataset/train/*/proj_depth/velodyne_raw/*/*.png"
```

Make sure that the regular expression is guarded by "".

### Set the environment variable $KITTIPATH
If you want to use the prebuilt datasets, you have to set the KITTIPATH variable.
Set it in your .bashrc file in your home directory or wherever you want.

The prebuilt dataset files have entries as follows:

$KITTIPATH/train/2011_10_03_drive_0042_sync/proj_depth/velodyne_raw/image_03/0000001161.png


## Training the networks

There are currently 2 example networks, that we also used for comparison in our paper. Currently, there is still a bit finetuning needed as we coded the paper in caffe and I'm currently struggeling to get the same results with tensorflow, but keep yourself updated on this page.

Both models can be trained by e.g.

```
./sparse_conv_net.py train
```

Up to this date, I have not implemented a test and eval only option, but they will be added in the future.

The logs are created in the repo folder, if you don't change the log_dir parameter in the experiment.

### Visualizing the results

You can visualize the results by using tensorboard. I added a rudimentary visualization.

```
tensorboard --logdir logs
```