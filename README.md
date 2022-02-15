# Fork of original AdaBins repository

Find the original repository [here](https://github.com/shariqfarooq123/AdaBins)

## Prerequisites

-   Make sure you have a CUDA-capable GPU (I've used an RTX 2060 on a laptop).
-   Make sure you have CUDA **10.1** installed.
-   Make sure you have `conda` installed.

## Preparation

```
### Replace with your directory of choice
export ADABINS_WORKSPACE=~
$ cd $ADABINS_WORKSPACE
### Make a folder for datasets
$ mkdir dataset
### Clone this repo
$ git clone https://github.com/ncsereoka/AdaBins.git
```

## Data

-   Don't use the dataset from DenseDepth! Details [here](https://github.com/shariqfarooq123/AdaBins/issues/54#issuecomment-1014929303).
-   Use the instructions from the [BTS repository](https://github.com/cleinc/bts/):

```
$ cd $ADABINS_WORKSPACE/AdaBins/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```

-   You now have the data! Onto the enironment setup.

## Environment

-   I've trained on an Arch Linux, so your environment might be slightly different.
-   Create a new Conda environment: `conda create -n adabins python=3.7` (using 3.7 but 3.8 works as well)
-   Activate the new environment `conda activate adabins`
-   Install packages using **pip** (had several issues with `conda install` so stuck with `pip`):
-   `pip install pytorch3d` (most annoying)
-   `pip install wandb`
-   `pip install matplotlib`
-   `pip install scipy`
-   At this point, when running the train command, you might run into the following issue:
    -   `ImportError: .../.conda/envs/adabins/lib/python3.7/site-packages/pytorch3d/_C.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZNK2at6Tensor7is_cudaEv`
    -   To fix this, uninstall `torch` and `torchvision`
    -   i.e. `pip uninstall torch torchvision`
    -   Run the command from [here](https://github.com/facebookresearch/maskrcnn-benchmark/issues/891#issuecomment-812496907)
    -   i.e. `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
    -   The import error should be no more.
-   Set up **wandb**:
    -   Either execute `wandb offline`
    -   Or: sign up on https://wandb.com,
    -   Go to your profile and copy your API token.
    -   Execute `wandb login`. Paste in your API token.
-   Your environment should be set up.

## Actual training

-   Start training by executing `python train.py args_train_nyu.txt` from the main directory.

## Other ssues you might encounter

-   Missing libraries (e.g. **libcudart**):
    -   Make sure you link them:
    -   `sudo ln -s /opt/cuda/targets/x86_64-linux/lib/libcudart.so.10.1 /usr/lib/libcudart.so.10.1`
-   Training stuck at 0%
    -   Double-check the text files which contain the filenames of the images. Some images might be missing (shouldn't happen with the steps from above).
-   GPU out of memory
    -   Reduce the batch size (I've reduced to the lowest batch size for it to fit my 6GB of GPU memory).
