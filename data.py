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
    images_train: np.ndarray | list
    images_test: np.ndarray | list
    labels_train: np.ndarray | list
    labels_test: np.ndarray | list

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
        self.images_train = []
        self.images_test = []
        self.labels_train = []
        self.labels_test = []

        
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

    def _type_size_conversion(self):
        """Convert images and labels from list to numpy array, and add dimensions for gray images."""
        self.images_train = np.array(self.images_train, np.float32)
        self.images_test = np.array(self.images_test, np.float32)
        self.labels_train = np.array(self.labels_train)
        self.labels_test = np.array(self.labels_test)

        if self.image_size[2] == 1:
            self.images_train = np.expand_dims(self.images_train, axis=-1)
            self.images_test = np.expand_dims(self.images_test, axis=-1)

    def _shuffle_data_sets(self):
        """Shuffle the training data set and test data set separately."""
        indices_train = tf.range(self.images_train.shape[0])
        indices_test = tf.range(self.images_test.shape[0])

        # Shuffle the indices
        shuffled_indices_train = tf.random.shuffle(indices_train)
        shuffled_indices_test = tf.random.shuffle(indices_test)

        # Shuffle the data sets
        self.images_train = self.images_train[shuffled_indices_train]
        self.labels_train = self.labels_train[shuffled_indices_train]
        self.images_test = self.images_test[shuffled_indices_test]
        self.labels_test = self.labels_test[shuffled_indices_test]

    def load(self, folder: str, shuffle: bool | None = True):
        """
        Load dataset from the folder and devide the data set to training set and test set.
        
        Args:
            folder (str): the folder containing the data set. The folder only has subfolders. \
                Each subfolder has the folder name as the class name and contains the images of this class.
            shuffle (bool): True indicates shuffle the training and test sets separately. Default to `True`.
        """

        if self.images_train:
            self.images_train = []
            self.images_test = []
            self.labels_train = []
            self.labels_test = []
            self.filenames_fail = []
            self.size_train = 0
            self.size_test = 0
            self.classes = {}

            warnings.warn('the DataLoader has loaded data before', RuntimeWarning)

        # for each label, generate train set and test set
        for label, dir in enumerate(os.listdir(folder)):
            self.classes[dir] = label
            filenames = os.listdir(os.path.join(folder, dir))

            # shuffle the filenames with same label, i.e., under the same folder
            if shuffle:
                filenames =[s.decode('utf-8') for s in tf.random.shuffle(filenames).numpy()]

            # index to split data sets
            index_split = int(self.ratio * len(filenames))

            # load training set
            for filename in filenames[0: index_split]:
                filename_full = os.path.join(folder, dir, filename)
                self._load_image_label(label=label ,filename=filename_full, images=self.images_train, labels=self.labels_train)
            # load test set
            for filename in filenames[index_split: ]:
                filename_full = os.path.join(folder, dir, filename)
                self._load_image_label(label=label ,filename=filename_full, images=self.images_test, labels=self.labels_test)
        
        self._type_size_conversion()
        if shuffle:
            self._shuffle_data_sets()

        if len(self.filenames_fail) !=0 :
            raise warnings.warn(f'{len(self.filenames_fail)} files were not successfully loaded', RuntimeWarning)
