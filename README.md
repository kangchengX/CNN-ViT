<p align="center">
    <h1 align="center">CNN-VIT</h1>
</p>
<p align="center">
    <em>CLT of CNN and ViT on a small data set.</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
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

This repo is a command line tool for image classification, with models VGG, ResNet, ViT, and MobileViT.

This project implements ViT and MobileViT from scratch, both of with are compatible with **graph execution mode** in tensorflow.

---

##  Repository Structure

```sh
└── CNN-ViT/
    └── data/
        ├── CQ
        │   ├── cq0.jpg
        │   ├── cq1.jpg
        │   └── ...
        ├── OTQ
        └── TQ
    ├── config.py
    ├── data.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── mobileViT.py
    │   └── models.py
    ├── requirements.txt
    └── setup.py
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

Some example images have already been placed in this folder.

### Data Sources
These example images data used in this project is collected from Palace Museum, Taipei, which contains 4 classes with 2000 around images of cultural relics.

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [config.py](config.py)               | Defines configurations for different architectures of the MobileViT model, allowing customization of parameters such as number of classes, image size, and dropout rates to adapt to various image classification tasks within the CNN-ViT projects architecture.                                                                              |
| [data.py](data.py)                   | DataLoader in CNN-ViT manages image dataset preprocessing by loading, normalizing, and partitioning data into training and testing sets, supporting image resizing and format adjustments for model compatibility, and including functionality for data shuffling.                                |
| [main.py](main.py)                   | `main.py` orchestrates the model training and evaluation pipeline including configuration, data loading, training, and evaluation of various neural network models including MobileViT, VGG, ResNet, and Vision Transformer.           |
| [requirements.txt](requirements.txt) | Contains the depandencies of the project.|
| [setup.py](setup.py)     | Some setup for this command line tool.|

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

> 1. Create a virtual environment:
>
> Windows:
> ```console
> py -3.10 -m venv cnn-vit-venv
> cnn-vit-venv\Scripts\activate
> ```
>
> Linux:
> ```console
> python3.10 -m venv cnn-vit-venv
> source cnn-vit-venv/bin/activate
> ```
>
> 2. Clone the repository:
>
> ```console
> git clone -b clt https://github.com/kangchengX/CNN-ViT.git
> ```
>
> 3. Change to the project directory:
> ```console
> cd CNN-ViT
> ```
>
> 4. Install the dependencies:
> ```console
> pip install -r requirements.txt
> ```
>
> 5. Build CLT:
> ```console
> python setup.py develop
> ```

### Data

Put the data folder with the structure described in the above section [Data](#data). Some example images are already placed in the `data` folder.

###  Usage

> ```console
> cnn-vit [config_arch] [OPTIONS]
> ```

**For help**:
> ```console
> cnn-vit --help
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
> ```console
> cnn-vit resnet50 --results_filename results
> ``` 

[**Return**](#overview)

---
