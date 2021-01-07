# VRDL HW4

This is homework 4 in NCTU Selected Topics in Visual Recognition using Deep Learning.

## Hardware

The following specs were used to create the original solution.

- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i7-6700 CPU @ 3.40 GHz
- NVIDIA GeForce GTX TITAN X

## Installation

1. Using Anaconda is strongly recommended. {envs_name} is the new environment name which you should assign.
    ```shell
    conda create -n {envs_name} python=3.7
    conda activate {envs_name}
    ```
2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,
    ```shell
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```
   Note: Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA
   version for precompiled packages on the [PyTorch website](https://pytorch.org/).

3. Use the following command to install other requirements.
    ```shell
    pip install -r requirements.txt
    ```

## Dataset Preparation

You can download the data [here](https://drive.google.com/drive/u/0/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x).

### Prepare Data and Code

After downloading and extracting, the data directory is structured as:

```text
+- data
    +- testing_lr_images
        +- 00.png
        +- 01.png
        ...
    +- training_hr_images
        +- 2092.png
        +- 8049.png
        ...
datasets.py
eval.py
model.py
test.py
train.py
train_val_split.py
utils.py
```

### Data Preprocessing

The following command is going to split the training data randomly by marking training data and validation data in two
directory in "data" directory, called "train" and "val", respectively. The ratio of the training data and validation
data is 8 : 2.

```shell
python3 train_val_split.py
```

## Training

The code not only trains, but also valid the model. You can train the model by following:

```shell
python3 train.py
```

## Evaluation

The code will evaluate the PSNR using the best checkpoint of the model, and compare with the bicubic image afterward.

```shell
python3 eval.py
```

## Testing

The code will create a folder called "output" which contains a few images. The images are super resolve from testing
images.

```shell
python3 test.py
```

## Acknowledgement

twtygqyy: [pytorch-SRResNet](https://github.com/twtygqyy/pytorch-SRResNet)

sgrvinod: [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution)


