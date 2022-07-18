"""Description
   -----------
This script includes all functions that are related to datasets, such as
data generators, preprocessing and augmentation

"""

from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from utils import data, augment
#from utils.augment import transforms


class InputNormalization:
    """
    Class containing preprocess_input options from keras.applications
    """

    def __init__(self, backbone_arch):
        """
        :param backbone_arch: name of the backbone architecture
        """
        self.backbone_arch = backbone_arch
        self.normalization_function = self.get_normalization_function()

    @staticmethod
    def preprocess_input_tf(img):
        """
        Scale image between -1 and 1
        :param img: RGB image
        :return: normalised image
        """
        img = img / 127.5
        img = img - 1.
        return img

    @staticmethod
    def preprocess_input_torch(img):
        """
        Scale image between 0 and 1, and normalize with respect to the ImageNet dataset
        :param img: RGB image
        :return: normalised image
        """
        img = img / 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img[..., 0] = img[..., 0] - mean[0]
        img[..., 1] = img[..., 1] - mean[1]
        img[..., 2] = img[..., 2] - mean[2]
        img[..., 0] = img[..., 0] / std[0]
        img[..., 1] = img[..., 1] / std[1]
        img[..., 2] = img[..., 2] / std[2]
        return img

    @staticmethod
    def preprocess_input_caffe(img):
        """
        Convert image from RGB to BGR, and normalize with respect to the ImageNet dataset
        :param img: RGB image
        :return: normalised image
        """
        img = img[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        img[..., 0] = img[..., 0] - mean[0]
        img[..., 1] = img[..., 1] - mean[1]
        img[..., 2] = img[..., 2] - mean[2]
        return img

    @staticmethod
    def preprocess_input_darknet(img):
        """
        Scale image between 0 and 1
        :param img: RGB image
        :return: normalized image
        """
        img = img / 255.
        return img

    @staticmethod
    def preprocess_input_none(img):
        """
        No pre-processing needed. Pre-processing is embedded in the initial layers of the backbone
        :param img: RGB image
        :return: input image
        """
        return img

    def get_normalization_function(self):
        """
        Get normalization function based on the backbone architecture
        :return: normalization function
        """
        if self.backbone_arch in ['MobileNet', 'MobileNetV2', 'Xception', 'ResNet50V2', 'NASNetMobile']:
            return self.preprocess_input_tf
        if self.backbone_arch in ['ResNet50']:
            return self.preprocess_input_caffe
        if self.backbone_arch in ['DenseNet121', 'SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNeXt50']:
            return self.preprocess_input_torch

        if self.backbone_arch == 'DarkNet':
            return self.preprocess_input_darknet
        if self.backbone_arch in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                                  'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
                                  'ResNet18', 'ResNet34', 'ResNeXt50']:
            return self.preprocess_input_none

        raise ValueError("Backbone architecture not recognized when fetching the normalization function")


@dataclass
class DataPaths:
    """
    Data class holding training and validation data paths
    """

    def __init__(self, args):
        """
        :param args: Parsed arguments
        """

        self.images, self.gts, _ = data.get_classification_data_paths(data_path=args.indir, args=args)

        if args.valdata:
            self.valid_img, self.valid_gt, _ = data.get_classification_data_paths(data_path=args.valdata, args=args)
            self.train_img, self.train_gt = self.images, self.gts
        else:
            self.train_img, self.valid_img, self.train_gt, self.valid_gt = \
                train_test_split(self.images, self.gts, test_size=args.validation_split, random_state=42)


class ResizingMethods:
    """
    Class containing resizing methods
    """

    def __init__(self, resize_method_name):
        """
        :param resize_method_name: string containing the name of resizing method
        """
        if resize_method_name == 'center-crop':
            self.resize = self.center_crop_resize
        elif resize_method_name == 'padding':
            self.resize = self.pad_resize
        elif resize_method_name == 'downscale':
            self.resize = self.downscale_resize
        else:
            raise ValueError("Resize method name not recognized")

    @staticmethod
    def center_crop_resize(image, target_dim):
        """Resize an image while holding the aspect ratio constant"""
        height = image.shape[0]
        width = image.shape[1]
        aspect = width / height
        if aspect > 1:
            # crop the left and right edges:
            offset = (width - height) / 2
            resize = (0, height, offset, width - offset)
        elif aspect < 1:
            # crop the top and bottom edges:
            offset = (height - width) / 2
            resize = (offset, height - offset, 0, width)
        else:
            resize = (0, height, 0, width)
        crop_img = image[int(resize[0]):int(resize[1]), int(resize[2]):int(resize[3])]
        return cv2.resize(crop_img, (target_dim[1], target_dim[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def pad_resize(image, target_dim):
        """Resize an image while holding the aspect ratio constant"""
        height, width, _ = image.shape
        target_height, target_width = target_dim
        aspect = width / height
        target_aspect = target_width / target_height
        if aspect > target_aspect:
            # pad top and bottom
            hpad = round((target_height * width / target_width - height) / 2)
            vpad = 0
        elif aspect < target_aspect:
            # pad left and right
            hpad = 0
            vpad = round((target_width * height / target_height - width) / 2)
        else:
            hpad = 0
            vpad = 0
        pad_img = cv2.copyMakeBorder(image, hpad, hpad, vpad, vpad, cv2.BORDER_CONSTANT)
        return cv2.resize(pad_img, (target_dim[1], target_dim[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def downscale_resize(image, target_dim):
        """Resize an image using OpenCV native resizing"""
        return cv2.resize(image, (target_dim[1], target_dim[0]), interpolation=cv2.INTER_NEAREST)


def read_image_test(path, resize, dim, preprocess_input):
    """Read image and apply preprocessing for testing"""
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, dim)
    img = img.astype('float32')
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def read_image_show(path, resize, dim):
    """Read image and don't apply preprocessing"""
    if isinstance(path, str):
        img = cv2.imread(path)
    else:
        img = path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, dim)
    img = img.astype('uint8')
    return img


def read_mask_test(path, resize, dim, num_classes):
    """Read mask and scale for testing"""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = resize(mask, dim)
    if num_classes == 2:
        mask = mask / 255.0
        mask = mask > 0.5
    mask = mask.astype(np.int32)
    return mask


def read_label_test(path):
    """Read labels for testing multi-label classifier"""
    with open(path, 'r', encoding="utf8") as file:
        labels = file.readline()
    labels = np.fromstring(labels, dtype='int32', sep=',')
    return labels


class ClassificationDataset:
    # pylint: disable=too-many-instance-attributes
    """
    Class to create tf.data instance for classification dataset, parse images and ground truths and batch samples
    """

    def __init__(self, img_paths, gts, args, batch=None, split='train', trials=None):
        """
        :param img_paths: paths to target images
        :param gts: ground truths
        :param args: parsed auser arguments
        :param split: datasetset split: either train, test, or validation
        """
        self.num_classes = args.num_classes
        self.dim = args.dim
        self.num_labels = bool(args.num_labels)
        self.label_smoothing_flag = args.label_smoothing
        self.label_precision = tf.float16 if self.label_smoothing_flag else tf.int32
        self.preprocess_input = InputNormalization(args.backbone).normalization_function
        if trials is None:
            self.transforms = augment.generate_augmentation(args)
        else:
            self.transforms = augment.generate_augmentation(args, trials)
             
        if self.label_smoothing_flag:
            if self.num_classes == 2:
                self.label_smoothing = self.smooth_labels_binary
            else:
                self.label_smoothing = self.smooth_labels_multi_class
            # TODO implement label smoothing for multi-label classification. Apply new metric in binary case
        self.resize = ResizingMethods(args.resize).resize
        autotune = tf.data.experimental.AUTOTUNE
        self.dataset = tf.data.Dataset.from_tensor_slices((img_paths, gts))
        if split != 'test':
            self.dataset = self.dataset.shuffle(len(img_paths))
        self.dataset = self.dataset.map(self.tf_parse)
        if split in ('validation', 'test'):
            self.dataset = self.dataset.map(self.valid_preprocess, num_parallel_calls=autotune)
        else:
            self.dataset = self.dataset.map(self.augment, num_parallel_calls=autotune)
        if batch is None:
            self.dataset = self.dataset.batch(args.batch)
        else:
            self.dataset = self.dataset.batch(batch)
        self.dataset = self.dataset.prefetch(buffer_size=autotune)
        if split != 'test':
            self.dataset = self.dataset.repeat()

    def tf_parse(self, img, gt):
        """Read images. tf.numpy_function to wrap the numpy function around tensors"""

        def _parse(img, gt):
            image = self.read_image(img)
            return image, gt

        image, _ = tf.numpy_function(_parse, [img, gt], [tf.float32, tf.int32])
        return image, gt

    def read_image(self, path):
        """Read image and apply preprocessing for training"""
        path = path.decode()
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize(img, self.dim)
        img = img.astype('float32')
        img = self.preprocess_input(img)
        return img

    def augment(self, img, gt):
        """Apply transforms on images, and cast data for classification tasks"""

        def _aug_fn(image, gt):
            _image = {"image": image}
            aug_data = self.transforms(**_image)
            _aug_img = aug_data["image"]
            return _aug_img, gt

        aug_img, _ = tf.numpy_function(_aug_fn, [img, gt], [tf.float32, tf.int32])

        aug_img.set_shape([self.dim[0], self.dim[1], 3])
        if self.num_classes > 2 and not self.num_labels:
            gt = tf.one_hot(gt, self.num_classes, dtype=self.label_precision)

        if self.label_smoothing_flag:
            gt = tf.dtypes.cast(gt, self.label_precision)
            gt = self.label_smoothing(gt)
        return aug_img, gt

    def valid_preprocess(self, img, gt):
        """Set shape and cast of validation set independently since they are not being augmented"""
        img = tf.dtypes.cast(img, tf.float32)
        img.set_shape([self.dim[0], self.dim[1], 3])
        if self.num_classes > 2 and not self.num_labels:
            gt = tf.one_hot(gt, self.num_classes, dtype=self.label_precision)

        else:
            gt = tf.dtypes.cast(gt, self.label_precision)
        if self.label_smoothing_flag:
            gt = self.label_smoothing(gt)

        return img, gt

    @staticmethod
    def smooth_labels_multi_class(labels, factor=0.1):
        """Label smoothing function for multi class scenario"""
        labels *= (1 - factor)
        labels += (factor / labels.shape[0])
        return labels

    @staticmethod
    def smooth_labels_binary(label, factor=0.02):
        """Label smoothing function for binary scenario"""
        label = label * (1.0 - factor) + 0.5 * factor
        return label
