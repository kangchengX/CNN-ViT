import cv2
import os
import numpy as np
import warnings
import tensorflow as tf
from typing import Tuple, Dict


class DataLoader():
    """
    The DataLoader of the images.
    
    Attributes:
        images_size (tuple): (height, width, channels) for input images with same shape.
        ratio (float): the ratio for training set in the whole data set.
        classes (dict): keys are the class names, values are the coresponding numerical label values.
        size_train (int): number of the images in the loaded training set.
        size_test (int): number of the images in the loaded test set.
        images_train (ndarray | None): the loaded images in training set. Initial is `None`. After loaded, this will become ndarray of images with shape (`size_train`, `image_size[0]`, `image_size[1]`, `image_size[2]`).
        images_test (ndarray | None): the loaded images in test set. Initial is `None`. After loaded, this will become ndarray of images with shape (`size_train`, `image_size[0]`, `image_size[1]`, `image_size[2]`).
        labels_train (ndarray | None): the loaded labels in training set. Initial is `None`. After loaded, converted to an array with shape (`size_train`,).
        labels_test (ndarray | None): the loaded labels in test set. Initial is `None`. After loaded, converted to an array with shape (`size_test`,).
    """
    image_size: Tuple[int, int, int]
    ratio: float
    classes: Dict[str, int]
    size_train: int
    size_test: int
    images_train: np.ndarray | None
    images_test: np.ndarray | None
    labels_train: np.ndarray | None
    labels_test: np.ndarray | None

    def __init__(
            self, 
            image_size: int | Tuple[int, int, int] | None = 224, 
            ratio: float | None = 0.75
    ):
        """
        Initialize the model.
        
        Args:
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple. Default to `224`.
            ratio (float): the ratio for training set in the whole data set. Default to `0.75`.
        """
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size, 3)
        else:
            self.image_size = image_size

        self.ratio = ratio
        self.classes = {}

        # record the filenames of images that openCV failed to read
        self.filenames_fail = []

        # size of the training and test sets
        self.size_train = 0
        self.size_test = 0

        # data for training and test sets
        self.images_train = None
        self.images_test = None
        self.labels_train = None
        self.labels_test = None

        
    def _process_image(self, img: np.ndarray):
        """
        Convert the image to a (height, width, channels) array with dtype float32 and devided by 255.
        
        Args:
            img (ndarray): the image to process.
        
        Returns:
            img (ndarray): the normalized image. None if failed to read the image.
        """
    
        img = cv2.resize(img, self.image_size[0:2])
        img = img.astype(np.float32)/255.0

        return img
    
    def _load_image_label(self, label: int, filename: str, images: list, labels: list):
        """
        Load image and the label to the training or test set.
        
        Args:
            label (int): label of the image.
            filename (str): path of the image.
            images (list): list to load the images.
            labels (list): list to load labels.
        """

        # read image from the file to numpy
        if self.image_size[2] == 1:
            img = cv2.imread(filename,0)
        else:
            img = cv2.imread(filename)
        
        if img is None:
            self.filenames_fail.append(filename)
            return

        img = self._process_image(img)

        # load image to dataset
        images.append(img)
        labels.append(label)

    def _shuffle(self, images: np.ndarray, labels: np.ndarray):
        """
        Shuffle the training and test set sparately.
        
        Args:
            images (ndarray): ndarray of the images of the whole dataset.
            labels (ndarray): ndarray of the labels of the whole dataset.
        """

        indices = tf.range(images.shape[0])

        # Shuffle the indices
        shuffled_indices = tf.random.shuffle(indices)

        images = images[shuffled_indices]
        labels = labels[shuffled_indices]


    def load(self, folder: str, shuffle: bool | None = True):
        """
        Load dataset from the folder and devide the data set to training set and test set.
        
        Args:
            folder (str): the folder containing the data set. The folder only has subfolders. \
                Each subfolder has the folder name as the class name and contains the images of this class.
            shuffle (bool): True indicates shuffle the training and test sets separately. Default to `True`.
        """

        if self.images_train is not None:
            self.images_train = None
            self.images_test = None
            self.labels_train = None
            self.labels_test = None
            self.filenames_fail = []
            self.size_train = 0
            self.size_test = 0
            self.classes = {}

            warnings.warn('the DataLoader has loaded data before', RuntimeWarning)

        images = []
        labels = []

        # for each label, generate train set and test set
        for label, dir in enumerate(os.listdir(folder)):
            self.classes[dir] = label
            filenames = os.listdir(os.path.join(folder, dir))

            # load training set
            for filename in filenames:
                filename_full = os.path.join(folder, dir, filename)
                self._load_image_label(label=label ,filename=filename_full, images=images, labels=labels)

        images = np.array(images, dtype=np.float32)
        if self.image_size[2] == 1:
            images = np.expand_dims(images, axis=-1)
        labels = np.array(labels)
        
        # shuffle images
        if shuffle:
            self._shuffle(images, labels)

        # split dataset to training and test
        size = images.shape[0]
        size_train = int(self.ratio * size)

        self.size_train = size_train
        self.size_test = size - self.size_train

        self.images_train = images[:self.size_train]
        self.images_test = images[self.size_train:]

        self.labels_train = labels[:self.size_train]
        self.labels_test = labels[self.size_train:]

        if len(self.filenames_fail) !=0 :
            raise warnings.warn(f'{len(self.filenames_fail)} files were not successfully loaded', RuntimeWarning)
