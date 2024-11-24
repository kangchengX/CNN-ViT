<p align="center">
    <h1 align="center">CNN-VIT</h1>
</p>
<p align="center">
    <em>Comparison of CNN and ViT on a small data set.</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
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
   - [Data Sources](#data-sources)
- [Modules](#modules)
- [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Data](#data-1)
   - [Usage](#usage)
      - [Run all the experiments](#run-all-the-experiments)
      - [Run the single pipeline](#run-the-single-pipeline)
</details>
<hr>

##  Overview

The CNN-ViT project compares multiple cnn and transformer-based models including VGG, ResNet, ViT and MobileViT on a small data set with 2 thousand images. We found that 1, ViT models' performance on this small data set is bad, even no more than 50%, whatever the structure is; 2, MobileViT could achieve similar accuracies but with only 10% around parameters compared with CNN models. 

This project implements ViT and MobileViT from scratch, both of with are compatible with **graph execution mode** in tensorflow.

---

##  Repository Structure

```sh
└── CNN-ViT/
    ├── config.py
    ├── data.py
    ├── experiments.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── mobileViT.py
    │   └── models.py
    └── requirements.txt
```
---


## Data

### Data Structure
The data folder needs to have the following structure and puts in the project folder if not specified in running:
```sh
└── data/
    ├── classname1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── classname2
    └── ...
```
, i.e., the `data` folder has child folders with class names as names, each of which contains the images of the correponding class.

### Data Sources
The data used in this project is collected from Palace Museum, Taipei, which contains 4 classes with 2000 around images of cultural relics.

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [config.py](config.py)               | Defines configurations for different architectures of the MobileViT model, allowing customization of parameters such as number of classes, image size, and dropout rates to adapt to various image classification tasks within the CNN-ViT projects architecture.                                                                              |
| [data.py](data.py)                   | DataLoader in CNN-ViT manages image dataset preprocessing by loading, normalizing, and partitioning data into training and testing sets, supporting image resizing and format adjustments for model compatibility, and including functionality for data shuffling.                                |
| [experiments.py](experiments.py)     | `experiments.py` studied two group of experients  - 1, Classification performance of VGG, ResNet and MobileViT with different architectures on the data set. 2, Classification performave of ViT with different architectures on this small data set. This scirpt handles configuration setup, execution, and result storage.                                         |
| [main.py](main.py)                   | `main.py` orchestrates the model training and evaluation pipeline including configuration, data loading, training, and evaluation of various neural network models including MobileViT, VGG, ResNet, and Vision Transformer.           |
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

**System Requirements:**

* **Python**: `3.10.4`

### Installation

<h4>From <code>source</code></h4>

> 1. Clone the repository:
>
> ```console
> $ git clone https://github.com/kangchengX/CNN-ViT.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd CNN-ViT
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

### Data

Put the data folder with the structure described in the above section [Data](#data).

###  Usage

#### Run all the experiments

Run [experiments.py](experiments.py).

#### Run the single pipeline

<h4>From <code>source</code></h4>

> ```console
> $ python main.py [config_arch] [OPTIONS]
> ```

**Command Line Arguments**:

| Argument | Type | Description | Default Value |
|--------|------|-------------|---------------|
|**Positional** |      |             |               |
| `config_arch`     | String | Architecture of the model. Value should be `mobilevit_xxs`, `mobilevit_xs`, `mobilevit_s`, `vgg16`, `vgg19`, `resnet50`, `resnet101`, or `vit`. | N/A |
|**Option** |      |             |               |
| `--image_size`     | Integer | Height or width of the input image. | `224` |
| `--image_channels` | Integer | Channels of the input image. | `3` |
| `--dropout`        | Float   | Patch size for the Vision Transformer. | `2` |
| `--vit_dim`        | Integer | (Word) dimension of the Vision Transformer. | `768` |
| `--vit_depth`      | Integer | Number of layers in the Vision Transformer. | `None`  |
| `--vit_num_heads`  | Integer | Number of attention heads in the Vision Transformer.  | `None` |
| `--vit_mlp_dim`    | Integer | Dimension of the MLP hidden layer in the Vision Transformer. | `1536` |
| `--split_ratio`    | Float   | Ratio of the training set in the whole dataset. | `0.75` |
| `--data_folder`    | String  | Folder containing the data.  | `data` |
| `--not_shuffle`    | Boolean | Indicates if to shuffle the dataset. (Set to `Flase` if present) | `True` |
| `--num_epochs`     | Integer | Number of epochs. | `20` |
| `--batch_size`     | Integer | Batch size. | `16` |
| `--learning_rate`  | Float | Learning rate. | `0.01` |
| `--results_filename`| String | Path to save the results. | `results` |

**Example**:
```console
$ python main.py resnet50 --results_filename results
``` 

[**Return**](#overview)

---
