"""Description
   -----------
This script includes losses and metrics methods used for training and evaluation
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike

class CustomMccMetric(tf.keras.metrics.Metric):
    """Computes the Matthews Correlation Coefficient.
    The statistic is also known as the phi coefficient.
    The Matthews correlation coefficient (MCC) is used in
    machine learning as a measure of the quality of binary
    and multiclass classifications. It takes into account
    true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even
    if the classes are of very different sizes. The correlation
    coefficient value of MCC is between -1 and +1. A
    coefficient of +1 represents a perfect prediction,
    0 an average random prediction and -1 an inverse
    prediction. The statistic is also known as
    the phi coefficient.
    MCC = (TP * TN) - (FP * FN) /
          ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)
    Args:
        num_classes : Number of unique classes in the dataset.
        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.
    """
    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        name: str = "CustomMccMetric",
        dtype: AcceptableDTypes = None,
        **kwargs,
    ):
        """Creates a Custom Matthews Correlation Coefficient instance."""
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        # self.TN = tf.Variable(0.)
        # self.TP = tf.Variable(0.)
        # self.FN = tf.Variable(0.)
        # self.FP = tf.Variable(0.)

        self.TN = self.add_weight(name='tn', initializer=tf.keras.initializers.zeros)
        self.TP = self.add_weight(name='tp', initializer=tf.keras.initializers.zeros)
        self.FN = self.add_weight(name='fn', initializer=tf.keras.initializers.zeros)
        self.FP = self.add_weight(name='fp', initializer=tf.keras.initializers.zeros)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(tf.math.round(y_pred), dtype=self.dtype)

        # Count true positives, true negatives, false positives and false negatives.
        value_TP = tf.math.count_nonzero(y_pred * y_true)
        value_TN = tf.math.count_nonzero((y_pred - 1) * (y_true - 1))
        value_FP = tf.math.count_nonzero(y_pred * (y_true - 1))
        value_FN = tf.math.count_nonzero((y_pred - 1) * y_true)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            value_TP = tf.multiply(value_TP, sample_weight)
            value_TN = tf.multiply(value_TN, sample_weight)
            value_FP = tf.multiply(value_FP, sample_weight)
            value_FN = tf.multiply(value_FN, sample_weight)
        
        self.TP = tf.reduce_sum(value_TP)
        self.TN = tf.reduce_sum(value_TN)
        self.FP = tf.reduce_sum(value_FP)
        self.FN = tf.reduce_sum(value_FN)

        # print(f"y_true: {y_true}, y_pred: {y_pred}")
        # print(" ")
        # print(f"TP: {self.TP}")
        # print(f"TN: {self.TN}")
        # print(f"FP: {self.FP}")
        # print(f"FN: {self.FN}")
    
    def result(self):

        #Calculate MCC
        # MCC = (TP * TN) - (FP * FN) /
        #   ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)

        Zahler = (self.TP * self.TN) - (self.FP * self.FN)
        Zahler = tf.cast(Zahler, tf.float32)
        Nenner = (self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP ) * (self.TN + self.FN)
        Nenner = tf.cast(Nenner, tf.float32)

        MCC = tf.divide(Zahler, tf.math.sqrt(Nenner))
        
        if tf.math.is_nan(MCC):
            MCC = tf.constant(0, dtype=self.dtype)
        
        return MCC

    def reset_state(self):

        self.TN = tf.Variable(0.)
        self.TP = tf.Variable(0.)
        self.FN = tf.Variable(0.)
        self.FP = tf.Variable(0.)

def get_metrics(num_classes, num_labels=0):
    if num_labels > 0:
        return [
            tfa.metrics.F1Score(num_classes=num_labels, average="micro", threshold=0.5, name='f1_micro'),
            tfa.metrics.F1Score(num_classes=num_labels, average="macro", threshold=0.5, name='f1_macro'),
            tfa.metrics.F1Score(num_classes=num_labels, average="weighted", threshold=0.5, name='f1_weighted')
        ]
    elif num_classes > 2:
        return ['accuracy',
                tfa.metrics.F1Score(num_classes=num_classes, average="micro", name='f1_micro'),
                tfa.metrics.F1Score(num_classes=num_classes, average="macro", name='f1_macro'),
                tfa.metrics.F1Score(num_classes=num_classes, average="weighted", name='f1_weighted')]
    else:
        return ['accuracy', tfa.metrics.F1Score(num_classes=1, threshold=0.5, average="micro", name='f1_binary')]
        # return ['accuracy', tfa.metrics.MatthewsCorrelationCoefficient(num_classes=1, name = 'MatthewsCorrelationCoefficient') 
        #         ,CustomMccMetric(num_classes=num_classes)
        #         ]


def f1_micro_loss_seg(y_true, y_pred):
    """
    Calculate the dice loss between two tensors
    :param y_true: gt tensor
    :param y_pred: predicted tensor
    :return: dice loss
    """
    return 1.0 - f1_micro_seg(y_true, y_pred)


def f1_micro_test(y_true, y_pred):
    """
    Calculate the dice score between two numpy arrays
    :param y_true: gt array
    :param y_pred: pred array
    :return: dice score
    """
    smooth = 1e-15
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def f1_micro_seg(y_true, y_pred):
    """
    Calculate the dice score between two tensors
    :param y_true: gt tensor
    :param y_pred: predicted tensor
    :return:
    """
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def f1_micro(y_true, y_pred):
    """
    Calculate the dice score between two tensors
    :param y_true: gt tensor
    :param y_pred: predicted tensor
    :return:
    """
    smooth = 1e-15
    # y_pred=tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def bbox_iou(bboxes1, bboxes2):
    """
    Find IoU of two bounding boxes
    :param bboxes1: first bbox as tensor
    :param bboxes2: second bbox as tensor
    :return: IoU
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
    bboxes1_coor = tf.concat([
        bboxes1[..., :2] - bboxes1[..., 2:] * 0.5, bboxes1[..., :2] + bboxes1[..., 2:] * 0.5, ],
        axis=-1, )
    bboxes2_coor = tf.concat([
        bboxes2[..., :2] - bboxes2[..., 2:] * 0.5, bboxes2[..., :2] + bboxes2[..., 2:] * 0.5, ],
        axis=-1, )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)
    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Find generalized IoU of two bounding boxes
    :param bboxes1: first bbox as tensor
    :param bboxes2: second bbox as tensor
    :return: GIoU
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat([
        bboxes1[..., :2] - bboxes1[..., 2:] * 0.5, bboxes1[..., :2] + bboxes1[..., 2:] * 0.5, ],
        axis=-1, )
    bboxes2_coor = tf.concat([
        bboxes2[..., :2] - bboxes2[..., 2:] * 0.5, bboxes2[..., :2] + bboxes2[..., 2:] * 0.5, ],
        axis=-1, )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area
    iou = tf.math.divide_no_nan(inter_area, union_area)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])
    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)
    return giou


def average_precision_voc(recall, precision):
    """
    Calculate the average precision according to VOC convention for detection of a single class,
    given recall and precision
    :param recall
    :param precision
    :return: average precision
    """
    recall.insert(0, 0.0)  # insert 0.0 at begining of list
    recall.append(1.0)  # insert 1.0 at end of list
    mrec = recall[:]
    precision.insert(0, 0.0)  # insert 0.0 at begining of list
    precision.append(0.0)  # insert 0.0 at end of list
    mpre = precision[:]

    # Make the precision monotonically decreasing (goes from the end to the beginning)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Create a list of indexes where the recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1

    # The Average Precision (AP) is the area under the curve
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def yolo_loss(num_classes, output_layers, strides, iou_loss_threshold):
    def loss(y_true, y_pred):
        """
        Calculate the total loss for yolo training
        :param y_true: gt as tensor
        :param y_pred: prediction as tensor
        :return: GIoU, confidence loss, probability loss
        """
        giou_loss_l = conf_loss_l = prob_loss_l = 0
        for i in range(len(output_layers)):
            conv, pred = y_pred[i * 2], y_pred[i * 2 + 1]
            conv_shape = tf.shape(conv)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]
            input_size = strides[i] * output_size
            conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_classes))

            conv_raw_conf = conv[:, :, :, :, 4:5]
            conv_raw_prob = conv[:, :, :, :, 5:]

            pred_xywh = pred[:, :, :, :, 0:4]
            pred_conf = pred[:, :, :, :, 4:5]

            label_xywh = y_true[i][0][:, :, :, :, 0:4]
            respond_bbox = y_true[i][0][:, :, :, :, 4:5]
            label_prob = y_true[i][0][:, :, :, :, 5:]

            giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
            input_size = tf.cast(input_size, tf.float32)

            bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
            giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

            iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                           y_true[i][1][:, np.newaxis, np.newaxis, np.newaxis, :, :])
            max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
            respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < iou_loss_threshold, tf.float32)
            conf_focal = tf.pow(respond_bbox - pred_conf, 2)

            conf_loss = conf_focal * (
                    respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                    +
                    respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            )

            prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

            giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
            prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

            giou_loss_l += giou_loss
            conf_loss_l += conf_loss
            prob_loss_l += prob_loss

        return giou_loss_l, conf_loss_l, prob_loss_l

    return loss

# def yolo_loss_single_sample(y_true, y_pred):
#     """
#     Calculate the total loss for yolo training for a single sample (suitable for model.fit)
#     Not used anywhere...
#     :param y_true: gt as tensor
#     :param y_pred: prediction as tensor
#     :return: GIoU, confidence loss, probability loss
#     :return:
#     """
#     giou_loss_l = conf_loss_l = prob_loss_l = 0
#     for i in range(len(args.output_layers)):
#         conv, pred = y_pred[i * 2], y_pred[i * 2 + 1]
#         conv_shape = tf.shape(conv)
#         output_size = conv_shape[0]
#         input_size = args.strides[i] * output_size
#         conv = tf.reshape(conv, (output_size, output_size, 3, 5 + args.num_classes))
#         # print(conv.shape)
#         conv_raw_conf = conv[:, :, :, 4:5]
#         conv_raw_prob = conv[:, :, :, 5:]
#
#         pred_xywh = pred[:, :, :, 0:4]
#         pred_conf = pred[:, :, :, 4:5]
#
#         label_xywh = y_true[i][0][:, :, :, 0:4]
#         respond_bbox = y_true[i][0][:, :, :, 4:5]
#         label_prob = y_true[i][0][:, :, :, 5:]
#
#         giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
#         input_size = tf.cast(input_size, tf.float32)
#
#         bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, 2:3] * label_xywh[:, :, :, 3:4] / (input_size ** 2)
#         giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
#
#         iou = bbox_iou(pred_xywh[:, :, :, np.newaxis, :], y_true[i][1][np.newaxis, np.newaxis, np.newaxis, :, :])
#         max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
#
#         respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < args.iou_loss_threshold, tf.float32)
#
#         conf_focal = tf.pow(respond_bbox - pred_conf, 2)
#
#         conf_loss = conf_focal * (
#                 respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
#                 +
#                 respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
#         )
#
#         prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
#
#         giou_loss = tf.reduce_sum(giou_loss)
#         conf_loss = tf.reduce_sum(conf_loss)
#         prob_loss = tf.reduce_sum(prob_loss)
#
#         giou_loss_l += giou_loss
#         conf_loss_l += conf_loss
#         prob_loss_l += prob_loss
#
#     return giou_loss_l, conf_loss_l, prob_loss_l
