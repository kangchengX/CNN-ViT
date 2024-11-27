<p align="center">
    <h1 align="center">CNN-VIT</h1>
</p>
<p align="center">
    <em>Docker for training CNN and ViT on a small data set.</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=Docker&logoColor=white" alt="Docker Badge">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
</p>

<br>
<details open>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
   - [Data Structure](#data-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
   - [Get the Image](#get-the-image)
   - [Run the Container](#run-the-container)
   - [Command Line Arguments](#command-line-arguments)
</details>
<hr>

##  Overview

This repo is a command line tool for image classification, with models VGG, ResNet, ViT, and MobileViT.

This project implements ViT and MobileViT from scratch, both of with are compatible with **graph execution mode** in tensorflow.

---

##  Repository Structure

```sh
└── CNN-ViT/
    ├── data/
    │   ├── CQ
    │   │   ├── cq0.jpg
    │   │   ├── cq1.jpg
    │   │   └── ...
    │   ├── OTQ
    │   └── TQ
    ├── config.py
    ├── data.py
    ├── Dockerfile
    ├── main.py
    ├── models/
    │   ├── __init__.py
    │   ├── mobileViT.py
    │   └── models.py
    └── requirements.txt
```
---


## Data

### Data Structure
The data folder must hold the following structure and is put in the project folder `CNN-ViT` if not specified in running:
```sh
└── data/
    ├── classname1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── classname2
    └── ...
```
, i.e., the data folder (in this case, the name of the folder is `data`) had child folders with class names as names, each of which contains the images of the correponding class.

Some example images have already been placed in this folder.

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [config.py](config.py)               | Defines configurations for different architectures of the MobileViT model, allowing customization of parameters such as number of classes, image size, and dropout rates to adapt to various image classification tasks within the CNN-ViT projects architecture.                                                                              |
| [data.py](data.py)                   | DataLoader in CNN-ViT manages image dataset preprocessing by loading, normalizing, and partitioning data into training and testing sets, supporting image resizing and format adjustments for model compatibility, and including functionality for data shuffling.                                |
| [Dockerfile](Dockerfile)     | Dockerfile to build the image. |
| [main.py](main.py)                   | `main.py` organizes the model training and evaluation pipeline including configuration, data loading, training, and evaluation of various neural network models including MobileViT, VGG, ResNet, and Vision Transformer.           |
| [requirements.txt](requirements.txt) | Contains the depandencies of the project.|

</details>

<details open><summary>models</summary>

| File                                 | Summary |
| ---                                  | --- |
| [mobileViT.py](models\mobileViT.py) | Integrates the MobileViT from scratch, including the rearranging of tensors. Compatible with graph execution mode. |
| [models.py](models\models.py)       | Introduces foundational components for building and operating complex neural network models including MLP and Transformer, alongside model architectures like VGG, ResNet,  and Vision Transformer (ViT) whcih is also implemented from scratch.   |

</details>

---

##  Getting Started

### Get the Image

<h4>Build from <code>source</code></h4>

> 1. Clone the repository:
>
> ```console
> git clone -b docker https://github.com/kangchengX/CNN-ViT.git
> ```
>
> 2. Build image:
>
> ```console
> docker build -t cnn-vit .
> ```

<h4>Or pull the image</h4>

> ```console
> docker pull kangchengx/cnn-vit:latest
> ```

### Run the Container

<h4>For help (default)</h4>

> ```console
> docker run cnn-vit
> ```

<h4>Test training a model</h4>

> ```console
> docker run cnn-vit resnet18 --num_epochs=1
> ```

<h4>Inspect results on host</h4>

> ```console
> docker run -v $(pwd)/results:/app/results cnn-vit resnet18 --num_epochs=1 --results_filename=results/results.json
> ```

Explanation: ```$(pwd)/results``` is the directory (absolute path) on the host, ```/app/results``` is the directory inside the container (absolute path). The work dir is ```/app```, and ```results/results.json``` is the relative path. Therefore, the absolute directory of the ```results.json``` is ```/app/results/```, which the directory on the host needs to be mapped to.

<h4>Inspect results on host and send data</h4>

> ```console
> docker run -v $(pwd)/results:/app/results -v $(pwd)/data:/app/data cnn-vit resnet18 --results_filename=results/results.json
> ```

Where the ```/data``` directory lies in the currect working directory on the host. It must has the same structure as described in [data](#data) section.

###  Command Line Arguments

All these are put after ```cnn-vit```.

| Argument | Type | Description | Default Value |
|--------|------|-------------|---------------|
|**Positional** |      |             |               |
| `config_arch`     | String | Architecture of the model. Value should be `mobilevit_xxs`, `mobilevit_xs`, `mobilevit_s`, `vgg16`, `vgg19`, `resnet50`, `resnet101`, or `vit`. | N/A |
|**Option** |      |             |               |
| `--image_size`     | Integer | Height or width of the input image. | `128` |
| `--image_channels` | Integer | Channels of the input image. | `3` |
| `--dropout`        | Float   | Dropout ratio. | `0.5` |
| `--vit_patch_size` | Integer | Patch size for the Vision Transformer. | `2` |
| `--vit_dim`        | Integer | (Word) dimension of the Vision Transformer. | `256` |
| `--vit_depth`      | Integer | Number of layers in the Vision Transformer. | `4`  |
| `--vit_num_heads`  | Integer | Number of attention heads in the Vision Transformer.  | `4` |
| `--vit_mlp_dim`    | Integer | Dimension of the MLP hidden layer in the Vision Transformer. | `512` |
| `--split_ratio`    | Float   | Ratio of the training set in the whole dataset. | `0.75` |
| `--data_folder`    | String  | Folder containing the data.  | `data` |
| `--not_shuffle`    | Boolean | Not to shuffle the dataset. If the value is `Flase`, i.e., present, the data will not be suffled. | `True` |
| `--num_epochs`     | Integer | Number of epochs. | `200` |
| `--batch_size`     | Integer | Batch size. | `16` |
| `--learning_rate`  | Float | Learning rate. | `1e-6` |
| `--results_filename`| String | Path to save the results. | `results` |

[**Return**](#overview)

---
