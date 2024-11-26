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
- [Results](#results)
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

The CNN-ViT project compares multiple cnn and transformer-based models including VGG, ResNet, ViT and MobileViT on a small data set with 2 thousand images. We found that 1, ViT' performance on this small data set is unfavourable with accuracy less 0.3; 2, MobileViT_xxs can improve accuracy but the performance is still worse than CNN.

This project implements ViT and MobileViT from scratch, both of with are compatible with **graph execution mode** in tensorflow.

---

##  Results
**On training data set:**
All the models are trained with the same learning rate, batch size and number of epochs
<img src="for-readme/losses and accuracies.png" alt="losses-accuracies-train">

**On test data set:** 
| model         | number of parameters | loss | accuracy | weighted accuracy | F1 score | confidence score |
|:--------------|---------------------:|-----:|---------:|------------------:|---------:|-----------------:|
| mobilevit_xxs |        9.52e+05      |0.992 |    0.570 |             0.548 |    0.551 |            0.597 |
| mobilevit_xs  |        1.93e+06      |1.360 |    0.312 |             0.250 |    0.119 |            0.313 |
| mobilevit_s   |        4.94e+06      |1.360 |    0.312 |             0.250 |    0.119 |            0.314 |
| vgg16         |        6.51e+07      |1.569 |    0.796 |             0.802 |    0.792 |            0.953 |
| vgg19         |        7.04e+07      |1.228 |    0.798 |             0.781 |    0.783 |            0.943 |
| resnet18      |        1.12e+07      |0.843 |    0.659 |             0.645 |    0.645 |            0.695 |
| resnet34      |        2.13e+07      |0.847 |    0.809 |             0.814 |    0.812 |            0.941 |
| vit           |        3.29e+06      |1.317 |    0.349 |             0.289 |    0.204 |            0.350 |

---

##  Repository Structure

```sh
└── CNN-ViT/
    ├── example-data/
    │   ├── CQ
    │   ├── QTQ
    │   ├── TQ
    │   └── YQ
    ├── for-readme/
    │   └── losses and accuracies.png
    ├── models/
    │   ├── __init__.py
    │   ├── mobileViT.py
    │   └── models.py
    ├── results/
    │   ├── results_not_vit,json
    │   └── results_vit.csv
    ├── config.py
    ├── data.py
    ├── experiments.py
    ├── main.py
    ├── README.md
    ├── requirements.txt
    └── test.py
```
---


## Data

### Data Structure
The data must hold the following structure and is put in the project folder `CNN-ViT` if not specified in running:
```sh
└── data/
    ├── classname1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── classname2
    └── ...
```
, i.e., the data folder, (in this case, it has a name `data`) has child folders with class names as names, each of which contains the images of the correponding class.

### Data Sources
The data used in this project is collected from Palace Museum, Taipei, which contains 4 classes with 2000 around images of cultural relics.

---

##  Modules

<details open><summary>.</summary>

| File / Directory                     | Summary |
| ---                                  | ------- |
| [config.py](config.py)               | Defines configurations for different architectures of the MobileViT model, allowing customization of parameters such as number of classes, image size, and dropout rates to adapt to various image classification tasks within the CNN-ViT projects architecture. |
| [data.py](data.py)                   | DataLoader in CNN-ViT manages image dataset preprocessing by loading, normalizing, and partitioning data into training and testing sets, supporting image resizing and format adjustments for model compatibility, and including functionality for data shuffling. |
| [experiments.py](experiments.py)     | `experiments.py` studied two group of experients  - 1, Classification performance of VGG, ResNet and MobileViT with different architectures on the data set. 2, Classification performave of ViT with different architectures on this small data set. This scirpt handles configuration setup, execution, and result storage. |
| [main.py](main.py)                   | `main.py` organizes the model training and evaluation pipeline including configuration, data loading, training, and evaluation of various neural network models including MobileViT, VGG, ResNet, and Vision Transformer. |
| [requirements.txt](requirements.txt) | Contains the depandencies of the project.|
| [test.py](test.py) | A script to test the repo. |
| [example-data](example-data/) | Contains example images. |
| [for-readme](for-readme/) | Contains support files for `README.md`. |
| [models](models/) | Contains scripts defining modules and models. See details below. |
| [results](results/) | Contains the results. |

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
> git clone https://github.com/kangchengX/CNN-ViT.git
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

### Data

Put the data folder with the structure described in the above section [Data](#data).

###  Usage

#### Run all the experiments

Run [experiments.py](experiments.py).

#### Run the single pipeline

<h4>From <code>source</code></h4>

> ```console
> python main.py [config_arch] [OPTIONS]
> ```

**Command Line Arguments**:

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

**Example**:
> ```console
> python main.py resnet50 --results_filename results
> ``` 

[**Return**](#overview)

---
