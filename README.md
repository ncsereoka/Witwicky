# Comparison of Monocular Depth Estimation Algorithms

Based on the original AdaBins repository from [here](https://github.com/shariqfarooq123/AdaBins)

## Prerequisites

I've trained on an Arch Linux with CUDA 11.6, so your environment might be slightly different.
-   Make sure you have a CUDA-capable GPU (I've used an RTX 2060 on a laptop).
-   Make sure you have CUDA installed.
-   Make sure you have `conda` installed.

## Preparation

```
### Replace with your directory of choice
export WITWICKY_WORKSPACE=~
$ cd $WITWICKY_WORKSPACE
### Make a folder for datasets
$ mkdir dataset
### Clone this repo
$ git clone https://github.com/ncsereoka/Witwicky.git
```

## Environment

- Create a new Conda environment: `conda create -n witwicky python=3.8`
- Activate the new environment `conda activate witwicky`
- Install packages using **pip** (had several issues with `conda install` so stuck with `pip`):
- Use the instructions from the official [PyTorch website](https://pytorch.org/) to install `torch` and others.
- For me, this meant: `pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
- We need to install `pytorch3d`:
  - Follow instructions from [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
  - For me:
  - `pip install fvcore iopath`
  - `pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1102/download.html`
- Install the remaining packages with: `pip install -r packages.txt`
- Set up **wandb**:
    -   Either execute `wandb offline` - or:
    -   Sign up on https://wandb.com,
    -   Go to your profile and copy your API token.
    -   Execute `wandb login`. Paste in your API token
- Your environment should be set up. Onto the data!

## Data

-   Don't use the dataset from DenseDepth! Details [here](https://github.com/shariqfarooq123/AdaBins/issues/54#issuecomment-1014929303).
-   Use the instructions from the [BTS repository](https://github.com/cleinc/bts/).
-   These commands will set you up with the test files

```
$ cd $WITWICKY_WORKSPACE/Witwicky/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```

-   However, we still need the training data. You'll find it using [this comment](https://github.com/cleinc/bts/issues/4#issuecomment-527120927).
-   To make it even clearer: use [this Google Drive link](https://drive.google.com/uc?id=1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP&export=download) and download `sync.zip` into `$WITWICKY_WORKSPACE/dataset/nyu_depth_v2`. After unzipping, you should have the `sync` folder, just like in Bhat's inputs' text file (`../dataset/nyu_depth_v2/sync/`).

-   You now have the data! Onto the actual training.

## Actual training

- Start training by executing `python train.py args_train_nyu_standard.txt` from the main directory.
- Use `python train.py args_train_nyu_dpt.txt` to train with the DPT backend.

## Other issues you might encounter

-   Training stuck at 0%
    -   Double-check the text files which contain the filenames of the images. Some images might be missing (shouldn't happen with the steps from above).
-   GPU out of memory
    -   Reduce the batch size (I've reduced to the lowest batch size for it to fit my 6GB of GPU memory).
