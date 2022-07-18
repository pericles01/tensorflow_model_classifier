"""Description
   -----------
This script includes utility functions which are related to datasets preparation and analytics
"""
from glob import glob
from os import walk, path, listdir
import numpy as np
import ntpath
import cv2


def dataset_stats(dir_name, dataset_size):
    """Calculate channel means and variances for a dataset
    for classification dir_name of all classes
    for segmentation dir_name for img/ only"""
    i = 0
    channel_means = np.zeros((3, dataset_size), dtype='float32')
    for root, dirs, files in walk(dir_name, topdown=False):
        for name in files:
            # the mean of each channel for all images
            img = cv2.imread(path.join(root, name), cv2.COLOR_BGR2RGB)
            channel_means[:, i] = np.mean(np.mean(img, axis=-3), axis=-2)
            i = i + 1
    mean = np.mean(channel_means, axis=-1).astype(int)
    # the variance across all channels
    variance = np.var(channel_means, axis=-1).astype(int)
    return mean, variance


def get_active_learning_data_paths(data_path, include_list, args):
    """
    This function extracts image file paths and their corresponding classes in a directory for classification.
    the directory can either contain subdirectories (gt given) or files (no gt given)
    :param data_path: path to frames
    :param include_list: list or path to txt-file with files to be included in training
    :return: image file paths, gt paths (if given), class names or indices
    """

    def _read_gt_multi_label(path):
        """Read labels for multi-label classifier"""
        with open(path) as fd:
            labels = fd.readline()
        labels = np.fromstring(labels, dtype='int32', sep=',')
        return labels

    img_paths = []
    gts = []
    class_names = []
    filepath = path.join(data_path, listdir(data_path)[0])

    # Multi label classification task
    if args.num_labels > 0:
        img_paths = sorted([path.join(data_path, file) for file in include_list])
        gts = sorted([img_path[:-4] + '.txt' for img_path in img_paths])

        if gts:
            for i in range(len(gts)):
                gts[i] = _read_gt_multi_label(gts[i])
        if not args.class_names:
            class_names = [str(n) for n in list(range(args.num_labels))]
        else:
            class_names = args.class_names
    # if data path contains directories, i.e. known classes
    elif not path.isfile(filepath):
        class_counter = 0
        class_dirs = sorted(listdir(data_path))
        class_names = class_dirs
        for class_dir in class_dirs:
            files = sorted(listdir(path.join(data_path, class_dir)))
            for file in files:
                if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.bmp'):
                    if file in include_list:
                        img_paths.append(path.join(data_path, class_dir, file))
                        gts.append(class_counter)
            class_counter += 1
        # Assert that class names are names in this scheme: 00_class, 01_class, ...
        for c in class_names:
            assert len(
                c.split("_")[0]) == 2, "Class names must start with double digit numbers followed by an underscore"
        if args.check_data:
            img_paths, gts = check_data(img_paths, gts, args)
        # Sort img_paths and classes according to data sample names to get the correct sequence
        img_paths, gts = zip(*sorted(zip(img_paths, gts), key=lambda x: ntpath.basename(x[0])))  # --> tuple
        img_paths, gts = (list(t) for t in (img_paths, gts))
    else:
        img_paths = sorted([path.join(data_path, file) for file in include_list])
        if not args.class_names:  # If class names not given, generate class names as numbers
            class_names = [str(n) for n in list(range(args.num_classes))]
        else:
            class_names = args.class_names
    return img_paths, gts, class_names


def get_classification_data_paths(data_path, args):
    """
    This function extracts image file paths and their corresponding classes in a directory for classification.
    the directory can either contain subdirectories (gt given) or files (no gt given)
    :param data_path
    :return: image file paths, gt paths (if given), class names or indices
    """

    def _read_gt_multi_label(path):
        """Read labels for multi-label classifier"""
        with open(path) as fd:
            labels = fd.readline()
        labels = np.fromstring(labels, dtype='int32', sep=',')
        return labels

    img_paths = []
    gts = []
    class_names = []
    filepath = path.join(data_path, listdir(data_path)[0])
    # Multi label classification task
    if args.num_labels > 0:
        img_paths = sorted([path.join(data_path, 'img+gt', i) for i in listdir(path.join(data_path, 'img+gt')) if
                            not i.endswith('.txt')])
        gts = sorted([path.join(data_path, 'img+gt', i) for i in listdir(path.join(data_path, 'img+gt')) if
                      i.endswith('.txt')])
        if gts:
            for i in range(len(gts)):
                gts[i] = _read_gt_multi_label(gts[i])
        if not args.class_names:
            class_names = [str(n) for n in list(range(args.num_labels))]
        else:
            class_names = args.class_names
    # if data path contains directories, i.e. known classes
    elif not path.isfile(filepath):
        class_counter = 0
        class_dirs = sorted(listdir(data_path))
        class_names = class_dirs
        for class_dir in class_dirs:
            files = sorted(listdir(path.join(data_path, class_dir)))
            for file in files:
                if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith('.bmp'):
                    img_paths.append(path.join(data_path, class_dir, file))
                    gts.append(class_counter)
            class_counter += 1
        # Assert that class names are names in this scheme: 00_class, 01_class, ...
        for c in class_names:
            assert len(
                c.split("_")[0]) == 2, "Class names must start with double digit numbers followed by an underscore"
        if args.check_data:
            img_paths, gts = check_data(img_paths, gts, args)
        # Sort img_paths and classes according to data sample names to get the correct sequence
        img_paths, gts = zip(*sorted(zip(img_paths, gts), key=lambda x: ntpath.basename(x[0])))  # --> tuple
        img_paths, gts = (list(t) for t in (img_paths, gts))
    else:
        img_paths = sorted([path.join(data_path, img) for img in listdir(data_path)])
        if not args.class_names:  # If class names not given, generate class names as numbers
            class_names = [str(n) for n in list(range(args.num_classes))]
        else:
            class_names = args.class_names
    return img_paths, gts, class_names


def check_data(image_paths, gt_paths, args):
    """
    Make sure that dataset doesnt contain corrupt files
    :param image_paths
    :param gt_paths
    :return: 
    """
    print("Checking data ...")
    images_not_processable_index = []
    for i in range(len(image_paths)):
        print('\r %d / %d' % (i + 1, len(image_paths)), end="")
        img = cv2.imread(image_paths[i])
        try:
            img.shape
        except:
            images_not_processable_index.append(i)

    if len(images_not_processable_index) > 0:
        print("\nImages that cannot be processed were found and excluded:")
        [print(image_paths[i]) for i in images_not_processable_index]
        [image_paths.pop(i) for i in reversed(images_not_processable_index)]
        if gt_paths:
            [gt_paths.pop(i) for i in reversed(images_not_processable_index)]
    else:
        print("\nAll images can be processed correctly")

    return image_paths, gt_paths
