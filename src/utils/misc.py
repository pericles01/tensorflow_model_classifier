"""Description
   -----------
This script includes miscellaneous utility functions
"""

from os import path, listdir

import cv2
import json
import numpy as np
import os
import pandas as pd
import timeit
import uuid


def create_config_file(args, save_path, validation_set, aug_transforms, num_train, num_valid,
                       start_time_stamp, end_time_stamp, k_means='only for detection', lr=None):
    """
    Generate an args.json file which contains information about the model and training parameters
    :param args: arguments parsed by argpar
    :param save_path: saved model destination path
    :param validation_set: list if validation samples' paths
    :param aug_transforms: list of used augmentations #  TODO for detection
    :param num_train: number of training samples
    :param num_valid: number of validation samples
    :param start_time_stamp: training start time
    :param end_time_stamp: training end time
    :param k_means: bool: k-means clustering used for anchor calculation (for detection)
    :param lr: learning rate(s)
    :return:
    """
    class_names = []
    if args.class_names is not None:
        class_names = args.class_names
    else:
        if args.num_labels:
            class_names = 'Not given'
        else:
            class_names = sorted(os.listdir(args.indir))

    if len(aug_transforms) == 0:
        aug_transforms = []
    else:
        aug_transforms = [str(tr) for tr in aug_transforms]

    num_classes = args.num_labels if args.num_labels else args.num_classes
    num_classes_str = "Number of labels" if args.num_labels else "Number of classes"
    class_names_str = "Label names" if args.num_labels else "Class names"

    json_content = {"Training and model parameters": {
        "UUID": str(uuid.uuid4()),
        "Commit hash": args.commit_hash,
        "Start timestamp": start_time_stamp,
        "End timestamp": end_time_stamp,
        num_classes_str: num_classes,
        "Backbone architecture": args.backbone,
        "Dataset path": args.indir,
        "Input image dimensions": args.dim,
        "Resizing method": args.resize,
        "Number of epochs": args.epochs,
        "Batch size": args.batch,
        "Learning rate": args.lr,
        "Learning rate decay": args.lr_decay,
        "Backbone training": args.train_backbone,
    }}

    if args.bayesian:
        json_content["Training and model parameters"]["Bayesian Network"] = True
        json_content["Training and model parameters"]["Bayesian Rate"] = args.bayesian

    json_content["Training and model parameters"]["Validation split"] = None if args.valdata else args.validation_split
    json_content["Training and model parameters"]["Number of training samples"] = num_train
    json_content["Training and model parameters"]["Number of validation samples"] = num_valid
    json_content["Training and model parameters"][class_names_str] = class_names
    json_content["Training and model parameters"]["Augmentation transforms"] = aug_transforms
    json_content["Training and model parameters"]["Validation set"] = validation_set

    with open(path.join(save_path, 'args.json'), 'w') as f:
        json.dump(json_content, f, indent=2)


def read_config_file(args):
    """
    Read args.json file and update args accordingly
    :param args: args from argparser
    :return: updated args according to args.json
    """
    with open(os.path.join(args.saved_model, 'args.json')) as json_file:
        json_content = json.load(json_file)

        args.backbone = json_content["Training and model parameters"]["Backbone architecture"]
        args.dim = json_content["Training and model parameters"]["Input image dimensions"]
        args.resize = json_content["Training and model parameters"]["Resizing method"]
        if "Number of labels" in json_content["Training and model parameters"]:
            args.num_labels = json_content["Training and model parameters"]["Number of labels"]
        else:
            args.num_classes = json_content["Training and model parameters"]["Number of classes"]

        return args


def create_model_fit_history_log_file(history_log, save_path):
    """
    Generate a .json file containing training callbacks log
    :param history_log: model.fit history variable
    :param save_path: path to save json file
    :return:
    """
    for keys in history_log:
        history_log[keys] = [float(x) for x in history_log[keys]]
    with open(save_path, 'w') as f:
        json.dump(history_log, f, indent=2)


def read_class_names(data_path):
    """
    Read class names for object detection data loader
    :param data_path
    :return: list of class names
    """
    names = {}
    with open(path.join(data_path, "class_names.txt"), 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def yolo2voc(bbox, img_shape):
    """
    Convert Yolo annotation format to PASCAL VOC. Used to calculate mAP
    :param bbox: input bounding box in Yolo annotation
    :param img_shape: input image dimensions
    :return: PASCAL VOC bounding box annotation format
    """
    img_height, img_width, _ = img_shape
    xmin = (bbox[0] - bbox[2] / 2) * img_width
    ymin = (bbox[1] - bbox[3] / 2) * img_height
    xmax = (bbox[0] + bbox[2] / 2) * img_width
    ymax = (bbox[1] + bbox[3] / 2) * img_height

    return xmin, ymin, xmax, ymax


def timing_analysis(func):
    """ returns the execution time (in seconds) for 100000 iterations of func"""
    return timeit.timeit(lambda: func, number=100000)


def overlay_text(image, raw_output, num_classes, num_labels, label_filtered, gt, margin=20, color=(255, 255, 255),
                 fontface=cv2.FONT_HERSHEY_DUPLEX, fontscale=0.5, thickness=1, threshold=.5):
    # text input as text-array?
    def _text_background(image, text, xmargin, ymargin, color, fontface=cv2.FONT_HERSHEY_DUPLEX, fontscale=0.5,
                         thickness=1):
        """puts the desired text on the image with a black background"""
        text_size, baseline = cv2.getTextSize(text, fontface, fontscale, thickness)
        cv2.rectangle(image, (xmargin, image.shape[0] - ymargin + baseline),
                      (xmargin + text_size[0], image.shape[0] - ymargin - text_size[1] - baseline), color=(0, 0, 0),
                      thickness=cv2.FILLED)
        cv2.putText(image, text, tuple((xmargin, image.shape[0] - ymargin)),
                    color=color, fontFace=fontface, fontScale=fontscale, thickness=thickness)

    if num_labels:
        raw_output_str = str(raw_output)
        label_str = str((raw_output > threshold).astype(int))
    elif num_classes == 2:
        raw_output_str = str(np.round(raw_output.item(), 3)) if raw_output > threshold else str(
            np.round(1 - raw_output.item(), 3))
        label_str = str(1) if raw_output > threshold else str(0)
    else:
        raw_output_str = str(raw_output)
        label_str = str(np.argmax(raw_output))

    _text_background(image, 'Raw Output: ' + raw_output_str, margin, margin + 20, color,
                     fontface, fontscale, thickness)
    _text_background(image, 'Label: ' + label_str, margin, margin + 40, color,
                     fontface, fontscale, thickness)
    if gt != []:
        _text_background(image, 'Ground Truth: ' + str(gt), margin, margin + 60, color,
                         fontface, fontscale, thickness)
    if label_filtered != None:
        _text_background(image, 'Label Filtered: ' + str(label_filtered), margin, margin, color,
                         fontface, fontscale, thickness)
    return image


def get_delimiter(csv_path):
    """
    Finds delimiter of a table format file
    :param csv_path: path to file
    :return: delimiter
    """
    if csv_path.lower().endswith(".csv"):
        return ","
    elif csv_path.lower().endswith(".tsv"):
        return "\t"
    else:
        print("Unknown file ending. Trying to load as csv")
        return ","


def read_csv_table(data_path, args):
    """
    Extract the most relevant information from tables for postprocessing and plotting
    :param data_path: path to csv file or the directory containing it
    :return: table content in a data frame, list of headers, list of image paths, list of gts, list of predictions,
     list of predictions' probabilities, list of class names
    """
    if not path.isfile(data_path):
        file_path = listdir(data_path)[0]  # currently accepting one file, to avoid confusion of resulting files
        data_frame = pd.read_csv(path.join(data_path, file_path), sep=get_delimiter(file_path), header=0)
    else:
        data_frame = pd.read_csv(data_path, sep=get_delimiter(data_path), header=0)
    data_frame.sort_values(by=['Name'], inplace=True)
    imgs = data_frame['Name']  # never used
    if 'Ground Truth' in data_frame:
        gts = data_frame['Ground Truth']
    else:
        gts = []
    labels = data_frame['Label']
    classes = np.unique(labels)
    # maxprob = data_frame['Probability'] # never used
    raw_output = data_frame['Raw Output']
    for idx in range(len(raw_output)):
        if args.num_labels:
            raw_output[idx] = np.asarray(list(filter(None, raw_output[idx].split(' '))), dtype=float)
            labels[idx] = np.asarray(list(filter(None, labels[idx].split(' '))), dtype=float)
            if len(gts) != 0:
                gts[idx] = np.asarray(list(filter(None, gts[idx].split(' '))), dtype=float)
        elif args.num_classes > 2:
            raw_output[idx] = np.asarray(list(filter(None, raw_output[idx].split(' '))), dtype=float)

    headers = list(data_frame)

    return data_frame, headers, imgs, gts, labels, raw_output, classes


def get_demo_predictions(preds_path):
    """
    Read a demo output file containing frames' numbers and predictions
    :param preds_path: path to the csv file containing demo predictions
    :return: frames' numbers and predictions
    """
    preds = []
    frames = []
    with open(preds_path) as fd:
        lines = fd.readlines()[1:]
        for line in lines:
            line = line.strip()
            line = line.split(',')
            frames.append(line[0])
            preds.append([float(val) for val in line[1:]])
    return frames, preds


class YoloKmeans:
    def __init__(self, cluster_number, anchors, data_path, model_image_size):
        """
        Estimate anchor sizes for a dataset using k-means clustering
        https://github.com/david8862/keras-YOLOv3-model-set/blob/master/tools/misc/kmeans.py
        :param cluster_number: number of anchors to be calculated. 9 for Yolo and 6 for YoloTiny
        :param anchors
        :param data_path
        :param model_image_size: image size used to train the model
        """
        self.cluster_number = cluster_number
        self.data_path = data_path
        self.anchors = anchors
        self.model_image_size = [model_image_size, model_image_size]
        sample_image = cv2.imread(os.path.join(data_path, 'img+bb',
                                               sorted(os.listdir(os.path.join(data_path, 'img+bb')))[0]))
        self.dataset_image_size = [sample_image.shape[1], sample_image.shape[0]]

    def _iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def _avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self._iou(boxes, clusters), axis=1)])
        return accuracy

    def _kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:
            distances = 1 - self._iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def _txt2boxes(self):
        files = sorted(
            [os.path.join(self.data_path, 'img+bb', i) for i in os.listdir(os.path.join(self.data_path, 'img+bb')) if
             i.endswith('.txt')])
        dataSet = []
        for file in files:
            with open(file) as fd:
                infos = fd.readlines()
                # get image size
                image_width, image_height = self.dataset_image_size
                length = len(infos)
                for i in range(length):
                    width = int((float(infos[i].split(" ")[3]) * image_width))
                    height = int((float(infos[i].split(" ")[4]) * image_height))
                    # rescale box size to model anchor size
                    scale = min(float(self.model_image_size[1]) / float(image_width),
                                float(self.model_image_size[0]) / float(image_height))
                    width = round(width * scale)
                    height = round(height * scale)
                    dataSet.append([width, height])
        result = np.array(dataSet)
        return result

    def _create_clusters(self):
        print("Calculating anchors using K-means clustering:")
        all_boxes = self._txt2boxes()
        result = self._kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}% \n".format(
            self._avg_iou(all_boxes, result) * 100))
        return np.reshape(result, self.anchors.shape)
