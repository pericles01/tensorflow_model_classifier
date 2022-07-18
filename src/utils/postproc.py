"""Description
   -----------
This script includes all functions that are related to postprocessing
"""

import tensorflow as tf
import numpy as np

from utils import misc
import argparse
import os
from datetime import datetime
from collections import deque

# from test import evaluate_binary_classification, evaluate_multi_class_classification, sort_classification_output, \
#     create_classification_predictions_table, classification_probability_overlay

# TODO train.py will raise error caused by tf.constant if gpus are not explicitly defined here
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TemporalFilter:
    def __init__(self, num_classes, num_labels, state_filter, filter_size, threshold=0.5, table=False):
        """ filter the predictions of a classification.

        :param num_classes: number of classes
        :param state_filter: the name of the dataset
        :param filter_size: length of buffer
        :param table: if applied on a table
        :returns: pred_class, pred_prob
        """
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.state_filter = state_filter
        self.filter_size = filter_size
        self.threshold = threshold
        self.table = table

        self.buffer_update = self._weighted_mean_update
        self.init_class = np.eye(num_classes)[0]

        if state_filter != '':
            self._buffer_init(self.filter_size, self.init_class)
            if state_filter == 'cholec80':
                self.transitions = [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6]]
                # TODO add phase transition lists for other datasets
            print("Temporal State filtering for %s transitions is used" % self.state_filter)
        else:
            self.buffer = deque(maxlen=filter_size)

        print("Temporal filtering with weighted-mean filter of size %d is used" % self.filter_size)

    def _filter_prediction(self, prediction, bin_seg=False):  # The main function of the filter
        if self.state_filter != '':
            update_buffer_flag = self._state_machine(prediction)
            if update_buffer_flag:
                return self.buffer_update(prediction, bin_seg)  # else return last prediction?
            else:
                return self.buffer_update(self.buffer[-1], bin_seg)
        else:
            return self.buffer_update(prediction, bin_seg)

    def _buffer_init(self, maxlen, init_class=0):
        """ creates a buffer of maxlen and fills it with init_class

        :param int maxlen: length of buffer
        :param int init_class: values for initialization, default = 0
        :return: initialized buffer
        """
        self.buffer = deque(maxlen=maxlen)
        for n in range(maxlen):
            self.buffer.append(init_class)

    # Buffer update methods
    def _weighted_mean_update(self, raw_output, bin_seg=False):
        self.buffer.append(raw_output)  # shape = (N,) for multiclass+multilabel, () for binary, float
        res = np.average(np.array(self.buffer), weights=np.arange(1, len(self.buffer) + 1), axis=0)
        if self.num_labels > 0:
            pred_classes = [1 if elem >= self.threshold else 0 for elem in res]
            return pred_classes, np.round(res, 3)
        elif self.num_classes == 2 and not self.table:
            if bin_seg:
                return res
            pred_class = 1 if res >= self.threshold else 0
            return pred_class, np.round(res, 3)
        else:
            return np.argmax(res), np.round(res, 3)

    # State transition methods
    def _state_machine(self, raw_output):
        next_elem = np.argmax(raw_output) if self.num_classes > 2 else raw_output
        previous_elem = np.argmax(self.buffer[-1])
        if next_elem not in self.transitions[previous_elem]:
            return False  # update_buffer?
        else:
            return True

        # commented out to not intervene with testing

    """else:  # probability
            if self.num_classes == 2:
                if prediction > 0.5 and prediction < self.state_filter[1]:
                    return False  # This is a placeholder
                elif prediction < 0.5 and (1 - prediction) < self.state_filter[0]:
                    return False  # This is a placeholder
            else:
                predpos = np.argmax(prediction)
                if prediction[predpos] < self.state_filter[predpos]:
                    return False  # This is a placeholder"""


def non_maximum_suppression(boxes, pred_conf, min_overlap, score_threshold, class_topk, num_classes):
    """ Apply native TF NMS algorithm on Yolo predictions as a final post-processing step"""
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=pred_conf,
        max_output_size_per_class=class_topk,
        max_total_size=num_classes * class_topk,
        iou_threshold=min_overlap,
        score_threshold=score_threshold
    )
    return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]


def table_temporal_filtering(data_path):
    # read csv
    print("Loading Table...")
    df, headers, _, _, labels, raw_output, classes = misc.read_csv_table(data_path, args)
    num_classes = len(classes)

    f = TemporalFilter(num_classes=num_classes, num_labels=args.num_labels,
                       state_filter=args.state_filter, filter_size=args.filter_size, table=True)
    label_filtered = np.zeros_like(labels)
    raw_filtered = np.zeros_like(raw_output)

    for i in range(len(raw_output)):
        print('\r Processing image %d / %d' % (i + 1, len(raw_output)), end="")
        label_filtered[i], raw_filtered[i] = f.buffer_update(raw_output[i])
    if 'Label Filtered' in headers:
        df = df.drop(columns='Label Filtered')
    if 'Raw Filtered' in headers:
        df = df.drop(columns='Raw Filtered')
    df.insert(headers.index('Raw Output') + 1, 'Label Filtered', label_filtered)
    df.insert(headers.index('Raw Output') + 2, 'Raw Filtered', raw_filtered)
    savepath = os.path.join(result_path, timeStamp, os.path.basename(data_path) + '.csv')
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    df.to_csv(savepath, sep=misc.get_delimiter(data_path), header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()
    
    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)
    
    from main_experiment import Exp
    args = Exp(conf)

    result_path = os.path.join(args.outdir, "results")
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)

    timeStamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #os.mkdir(os.path.join(result_path, timeStamp))
    print("Start: " + str(datetime.now()))

    if args.task == 'classification':
        table_temporal_filtering(args.indir)

    print("\nEnd: " + str(datetime.now()))
