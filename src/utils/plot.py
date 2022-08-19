"""Description
   -----------
This script includes functions associated with generating plots.
Currently supporting:
# Un- / normalized confusion matrix
# Segmentation certainty
# Detection bounding boxes drawing
# Classification sequence line plot
# Classification sequence probability plot
"""

import colorsys
import itertools
import os
import random
from textwrap import wrap
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import ntpath
from utils import misc
import argparse
from datetime import datetime
from azureml.core import Run


def confusion_matrix(cm_input, labels, savepath, title='Confusion matrix', normalize=True, cmap='Blues', local=False):
    """
    Create a confusion matrix plot
    :param cm_input: np.array of tps, tns, fps and fns
    :param labels: list of class names
    :param title: title of plot
    :param normalize: if true, normalize metric before plotting
    :param cmap: colormap of confusion matrix
    :return:
    """
    font_size = 14 + len(labels)
    if normalize:
        cm_input = cm_input.astype('float') / cm_input.sum(axis=1)[:, np.newaxis]
        cm_input = np.nan_to_num(cm_input)
    elif np.max(cm_input) > 1:
        cm_input = cm_input.astype(int)
    if isinstance(labels[0], str):
        ['\n'.join(wrap(label, 20, break_long_words=False)) for label in labels]
    norm = colors.Normalize(vmin=0, vmax=1) if normalize else None
    plt.figure(figsize=(2 * len(labels), 2 * len(labels)))
    plt.imshow(cm_input, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title('\n'.join(wrap(title, 12 * len(labels))), fontsize=font_size)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, rotation_mode="anchor", ha="right", fontsize=font_size)
    plt.yticks(tick_marks, labels, fontsize=font_size)

    thresh = np.max(cm_input) / 1.5 if normalize else np.max(cm_input) / 2
    for i, j in itertools.product(range(cm_input.shape[0]), range(cm_input.shape[1])):
        if normalize or np.max(cm_input) <= 1:
            plt.text(j, i, "{:0.3f}".format(cm_input[i, j]),
                     horizontalalignment="center",
                     color="white" if cm_input[i, j] > thresh else "black", fontsize=font_size, )
        else:
            plt.text(j, i, "{:d}".format(cm_input[i, j]), horizontalalignment="center",
                     color="white" if cm_input[i, j] > thresh else "black", fontsize=font_size * 2 / 3)
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)
    plt.tight_layout()
    if not local:
        run = Run.get_context()
        run.log_image("Confusion Matrix", plot=plt)
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()


def segmentation_heatmap(img, gt, pred, heatmap, save_path, num_classes):
    """
    Create a plot containing the original image, gt, prediction and pixel-wise certainty for binary/multi class segmentation
    :param img: sample input image
    :param gt: sample ground truth
    :param pred: model prediction
    :param heatmap: pixel-wise certainty as np.array
    :param save_path: path to results/time_stamp/image_name
    :param num_classes: number of classes
    :return:
    """
    if num_classes == 2:
        if gt is not None:
            fig, ax = plt.subplots(1, 4, figsize=(16, 9))
            ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Image')
            ax[1].imshow(np.squeeze(gt), 'gray')
            ax[1].set_title('Ground truth')
            ax[2].imshow(np.squeeze(pred), 'gray')
            ax[2].set_title('Prediction')
            ax[3].imshow(np.squeeze(heatmap), 'jet')
            ax[3].set_title('Heatmap')
        else:
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))
            ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[0].set_title('Image')
            ax[1].imshow(np.squeeze(pred), 'gray')
            ax[1].set_title('Prediction')
            ax[2].imshow(np.squeeze(heatmap), 'jet')
            ax[2].set_title('Heatmap')
    else:
        if gt is not None:
            fig, ax = plt.subplots(1, 4, figsize=(16, 9))
            ax[0].set_title('Image')
            ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Ground truth')
            ax[1].imshow(gt.astype('uint8'))
            ax[2].set_title('Prediction')
            ax[2].imshow(pred.astype('uint8'))
            ax[3].set_title('Heatmap')
            ax[3].imshow(heatmap, 'jet')
        else:
            fig, ax = plt.subplots(1, 3, figsize=(16, 9))
            ax[0].set_title('Image')
            ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[1].set_title('Prediction')
            ax[1].imshow(pred.astype('uint8'))
            ax[2].set_title('Heatmap')
            ax[2].imshow(heatmap, 'jet')
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def segmentation_heatmap_separate(pred, heatmap, save_path, num_classes):
    """
    Create separate plots containing predictions and pixel-wise certainties for binary/multi class segmentation
    :param pred: model prediction
    :param heatmap: pixel-wise certainty as np.array
    :param save_path: path to results/time_stamp/image_name
    :param num_classes: number of classes
    :return:
    """
    if not os.path.exists(os.path.join(os.path.dirname(save_path), 'pred')):
        os.makedirs(os.path.join(os.path.dirname(save_path), 'pred'))
        os.makedirs(os.path.join(os.path.dirname(save_path), 'heatmap'))

    if num_classes == 2:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_axis_off()
        ax.imshow(np.squeeze(pred), 'gray')
        plt.savefig(os.path.join(os.path.dirname(save_path), 'pred', ntpath.basename(save_path)), bbox_inches='tight')
        ax.imshow(np.squeeze(heatmap), 'jet')
        plt.savefig(os.path.join(os.path.dirname(save_path), 'heatmap', ntpath.basename(save_path)),
                    bbox_inches='tight')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        ax.set_axis_off()
        ax.imshow(pred.astype('uint8'))
        plt.savefig(os.path.join(os.path.dirname(save_path), 'pred', ntpath.basename(save_path)), bbox_inches='tight')
        ax.imshow(heatmap, 'jet')
        plt.savefig(os.path.join(os.path.dirname(save_path), 'heatmap', ntpath.basename(save_path)),
                    bbox_inches='tight')
    plt.close(fig)


def draw_detection_bounding_box(image, preds, data_path, gt, points_overlay=False, show_label=True):
    """
    Draw predicted bounding boxes around objects detected by a Yolo model
    :param image: original input image (without scaling)
    :param preds: predicted bounding boxes after being decoded and filtered
    :param data_path: path where class_names.txt can be found
    :param show_label: if True, the name of the class will be overlaid on top of each box
    :return: image with a bounding box and class name overlay
    """

    classes = {}
    with open(os.path.join(data_path, "class_names.txt"), 'r') as data:
        for ID, name in enumerate(data):
            classes[ID] = name.strip('\n')

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    if preds:
        out_boxes, out_scores, out_classes, num_boxes = preds
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)

            score = out_scores[0][i]
            class_ind = int(out_classes[0][i])
            bbox_color = colors[class_ind]
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            if points_overlay:
                cv2.circle(image, (int((coor[1] + coor[3]) / 2), int((coor[0] + coor[2]) / 2)), radius=5,
                           color=bbox_color, thickness=2)
                bbox_mess = '%.2f' % score
                cv2.putText(image, bbox_mess, (int(((coor[1] + coor[3]) / 2) + 10), int((coor[0] + coor[2]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, bbox_color, bbox_thick // 2, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                if show_label:
                    bbox_mess = '%s: %.2f' % (class_ind, score)
                    t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                    c3 = (c2[0] - t_size[0], c2[1] - t_size[1] - 3)
                    cv2.rectangle(image, c2, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled

                    cv2.putText(image, bbox_mess, (np.float32(c2[0] - t_size[0] + 2), c2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    if gt:
        for i in range(len(gt)):
            box = gt[i].split(' ')[1:]
            box = [float(b) for b in box]
            coor = [0] * 4
            coor[0] = box[1] - box[3] / 2
            coor[1] = box[0] - box[2] / 2
            coor[2] = box[1] + box[3] / 2
            coor[3] = box[0] + box[2] / 2
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)

            bbox_color = (255, 255, 255)
            c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
            if points_overlay:
                cv2.circle(image, (int((coor[1] + coor[3]) / 2), int((coor[0] + coor[2]) / 2)), radius=5,
                           color=bbox_color, thickness=2)
            else:
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                t_size = cv2.getTextSize(gt[i].split(' ')[0], 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] + t_size[1] + 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1)  # filled
                cv2.putText(image, gt[i].split(' ')[0], (c1[0], np.float32(c1[1] + t_size[1] + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


def show_detection_bounding_box(data_path):
    """
    Iterate over a detection dataset with given ground truth and generate the images with the bounding boxes overlayed
    :param data_path: original input image (without scaling)
    :return
    """
    if not os.path.exists(args.output_results):
        os.makedirs(args.output_results)
    timeStamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir(args.output_results + timeStamp)

    img_paths = sorted([os.path.join(data_path, 'img+bb', i) for i in os.listdir(os.path.join(data_path, 'img+bb')) if
                        not i.endswith('.txt')])
    gt_paths = sorted([os.path.join(data_path, 'img+bb', i) for i in os.listdir(os.path.join(data_path, 'img+bb')) if
                       i.endswith('.txt')])

    for i in range(len(img_paths)):
        print('\r Processing image %d / %d' % (i + 1, len(img_paths)), end="")
        img = cv2.imread(img_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with open(gt_paths[i]) as f:
            gt = f.readlines()
        f.close()

        image_out = draw_detection_bounding_box(img, None, args.data, gt, args.points)
        image_out = cv2.cvtColor(np.array(image_out), cv2.COLOR_BGR2RGB)
        cv2.imwrite('./results/' + timeStamp + '/' + os.path.basename(img_paths[i]), image_out)


def phase_output_plot(data_path):
    """
    Plots the frame number on x-axis and the raw model output on y-axis. The phases are shown in different colors, dots showing
    the prediction and bars showing the ground truth.
    :param data_path: path to the directory containing a csv table
    :return:
    """
    df, _, imgs, gts, _, _, class_names = misc.read_csv_table(data_path, args)
    #    data.sort_values(by=['Name'], inplace=True)
    pred_per_class = [[] for _ in range(len(class_names))]
    gt_per_class = [[] for _ in range(len(class_names))]

    pred_col_name = 'Label'
    prob_col_name = 'Raw Output'
    if args.temp_filter:
        pred_col_name = 'Label Filtered'
        prob_col_name = 'Raw Filtered'
    for class_idx in range(len(class_names)):
        for img_idx in range(len(imgs)):
            pred_class = df[pred_col_name][img_idx]
            gt_class = gts[img_idx]
            class_name = class_names[class_idx]
            if pred_class == class_name:
                pred_per_class[class_idx].append(np.max(df[prob_col_name][img_idx]))
            else:
                pred_per_class[class_idx].append(np.nan)
            if gt_class == class_name:
                gt_per_class[class_idx].append(1.0)
            else:
                gt_per_class[class_idx].append(np.nan)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for class_idx in range(len(class_names)):
        prob_gt_list = gt_per_class[class_idx]
        ax.bar(np.hstack(np.argwhere(~np.isnan(np.asarray(prob_gt_list)))), 1.0, alpha=.25, width=1.0)

        prob_pred_list = pred_per_class[class_idx]
        ax.plot(np.hstack(np.argwhere(~np.isnan(np.asarray(prob_pred_list)))),
                [x for x in prob_pred_list if str(x) != 'nan'], '.', label=class_names[class_idx])

    plt.ylabel('Raw Output Max')
    plt.xlabel('Frame')
    plt.title("Prediction Results")

    plt.legend()
    plt.show()


def sequence_classification_line_plot(data, num_classes, title="line plot", labels=None, subplot=False,
                                      colors=None, THRESHOLD=0.5):
    """
    Plots the frame number on x-axis and the predicted class on y-axis. The ground truth is shown as line, the prediction as crosses
    :param csv_path: csv file containing the prediction results on test data
    :param temp_filtering: bool is to use temp_filtering column
    :return:
    """

    if colors is None:
        colors = ['g', 'k', 'b']
    if labels is None:
        labels = ['Phase'] * len(data)
    if type(data[1][0]) == str:
        data = np.asarray([[int(val[:2]) for val in col] for col in data])

    x = np.arange(0, len(data[0]), 1)

    if subplot:
        fig = plt.figure()
        fig.subplots(nrows=len(data), ncols=1, sharex=True, gridspec_kw={'hspace': 0})
        for i, ax in enumerate(fig.axes):
            ax.plot(x, data[i], color=colors[i])
            ax.set_ylabel(labels[i])
            if num_classes == 2:
                ax.set_yticks([0, THRESHOLD, round(np.max(data[i]))])
            else:
                ax.set_yticks(np.arange(num_classes))
        plt.xlabel('Frame')
        fig.suptitle(title, fontsize=14)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if type(data[0]) != list:
            ax.plot(np.array(range(0, int(data[0].shape[0]))), data[0], '-', color='g', label='ground truth')
        else:
            labels.remove('Ground Truth')
        ax.plot(np.array(range(0, int(data[1].shape[0]))), data[1], 'x', color='b', alpha=.7, label='predicted')
        if data.shape[0] == 3:  # Filtered predictions exist:
            ax.plot(np.array(range(0, int(data[2].shape[0]))), data[2], 'x', color='r', alpha=.4, label='filtered')
        plt.ylabel('Phase')
        plt.xlabel('Frame')
        if num_classes == 2:
            plt.yticks([0, THRESHOLD, np.max(data)])
        else:
            plt.yticks(np.arange(num_classes))
        fig.legend(labels, loc='lower center', bbox_to_anchor=(0.1, 0.02, 0.9, 0.2), ncol=len(data),
                   mode="expand")
        plt.title(title, fontsize=14)
    fig.tight_layout()
    plt.show()


def sequence_classification_line_plot_v2(data, title="line plot", threshold=0.5):
    """
    Plots the frame number on x-axis and the predicted class on y-axis. The ground truth is shown as green circles,
    the raw model output as red crosses, raw filter output as dashed gray line, and final label after
    thresholding in black line .
    Currently supporting binary classification only
    :param data: a data list containing gt, raw output, filter raw and filtered label
    :param title:
    :param THRESHOLD:
    """
    fig = plt.figure()
    x = np.arange(0, len(data[0]), 1)
    plt.plot(x, data[0], '.', markersize=10, color='green', alpha=.5, label="Ground Truth")
    plt.plot(x, data[1], 'x', markersize=2, color='red', label="Raw Output")
    plt.plot(x, data[2], linewidth=0.7, color='black', alpha=.9, label="Label Filtered")
    plt.plot(x, data[3], linewidth=2, color='grey', alpha=.7, linestyle='--', dashes=(10, 1), label="Raw Filtered")
    plt.yticks([0, threshold, round(np.max(data[0]))])
    plt.xlabel('Frame')
    plt.title(title, fontsize=14)
    plt.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(data_path):
    """
    Calls confusion matrix plotting for a table containing classification predictions.
    Also supports filtered predictions
    :param data_path: path to the directory containing a csv table
    :return:
    """
    print("Loading Tables...")
    df, headers, _, gts, preds, _, classes = misc.read_csv_table(data_path, args)
    assert len(gts) > 0, "Table does not contain ground truths"

    if type(preds[0]) == str:
        preds = np.asarray([int(val[:2]) for val in preds])
        gts = np.asarray([int(val[:2]) for val in gts])

    if len(classes) == 2:
        tp = np.sum(np.logical_and(preds == 1, gts == 1))
        tn = np.sum(np.logical_and(preds == 0, gts == 0))
        fp = np.sum(np.logical_and(preds == 1, gts == 0))
        fn = np.sum(np.logical_and(preds == 0, gts == 1))
        plot_labels = classes
        confusion_matrix(cm_input=np.array([[tn, fp], [fn, tp]]), labels=plot_labels, savepath=None,
                         title='Confusion matrix', normalize=True)
        if 'Label Filtered' in headers:
            filtered = df['Label Filtered']
            if type(filtered[0]) == str:
                filtered = np.asarray([int(val[:2]) for val in filtered])
            tp = np.sum(np.logical_and(filtered == 1, gts == 1))
            tn = np.sum(np.logical_and(filtered == 0, gts == 0))
            fp = np.sum(np.logical_and(filtered == 1, gts == 0))
            fn = np.sum(np.logical_and(filtered == 0, gts == 1))
            confusion_matrix(cm_input=np.array([[tn, fp], [fn, tp]]), labels=plot_labels, savepath=None,
                             title='Confusion matrix Filtered', normalize=True)
    else:
        confusion_mat = np.zeros((len(classes), len(classes)))
        for j in range(len(classes)):
            for k in range(len(classes)):
                confusion_mat[j, k] = np.sum(np.logical_and(gts == j, preds == k))
        plot_labels = args.class_names if args.class_names else np.arange(len(classes))
        confusion_matrix(cm_input=confusion_mat, labels=plot_labels, savepath=None, normalize=True)
        if 'Label Filtered' in headers:
            filtered = df['Label Filtered']
            if type(filtered[0]) == str:
                filtered = np.asarray([int(val[:2]) for val in filtered])
            confusion_mat = np.zeros((len(classes), len(classes)))
            for j in range(len(classes)):
                for k in range(len(classes)):
                    confusion_mat[j, k] = np.sum(np.logical_and(gts == j, filtered == k))
            plot_labels = args.class_names if args.class_names else np.arange(len(classes))
            confusion_matrix(cm_input=confusion_mat, labels=plot_labels, savepath=None,
                             title='Confusion matrix Filtered',
                             normalize=True)


def plot_sequence_classification_line_plot(data_path):
    """
    Calls sequence classification line plot for a table containing classification predictions.
    Also supports filtered predictions
    :param data_path: path to the directory containing a csv table
    :return:
    """
    df, _, _, gts, preds, _, classes = misc.read_csv_table(data_path, args)
    if 'Label Filtered' in df:
        filtered = df['Label Filtered']
        sequence_classification_line_plot(np.array([gts, preds, filtered]), len(classes),
                                          title="Line plot",
                                          labels=["Ground Truth", "Label", "Label Filtered"],
                                          subplot=False)
    else:
        sequence_classification_line_plot(np.array([gts, preds]), len(classes),
                                          title="Line plot",
                                          labels=["Ground Truth", "Label"],
                                          subplot=False)


def plot_img_attributions(image_show, savepath, prediction, gt, attribution_pred_mask, baseline_type, overlay_alpha,
                          labels=None, clip=True, cmap=None):
    """ This function plots the image together with the attribution mask + overlay

    :param image_show: image-array that will be shown on the plot
    :param savepath: path to save the plot
    :param prediction: the prediction of the model for the image, will be printed below the original image
    :param gt: the gt-array of the image, will be printed below the original image
    :param attribution_pred_mask: heatmap sowing attributions of image
    :param baseline_type: name of interpretability method
    :param overlay_alpha: alpha value for overlay of heatmap and image
    :param labels: positively predicted labels in a multi-label-classification task
    :param clip: activate clipping of mask on 99th percentile
    :param cmap: colormap for heatmap
    :return:
    """
    if labels is not None:
        fig, axs = plt.subplots(nrows=len(labels), ncols=3, squeeze=False, figsize=(12, len(labels) * 4))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(12, 4))

    axs[0, 0].set_title('Original image')
    axs[0, 0].imshow(image_show)
    axs[0, 0].axis('off')
    axs[0, 0].annotate('Prediction: ' + str(np.round(prediction, 2)), xy=(1, -0.06), xycoords='axes fraction',
                       ha='right', va='bottom')
    if gt is not None:
        axs[0, 0].annotate('GT: ' + str(gt), xy=(1, -0.11), xycoords='axes fraction', ha='right', va='bottom')

    axs[0, 1].set_title('Attribution mask (Pred) - ' + baseline_type)
    axs[0, 2].set_title('Overlay (Pred)')

    if labels:  # For multi-label classification
        for i in range(len(labels)):
            vmax = np.percentile(attribution_pred_mask[i], 99) if clip else 255
            axs[i, 0].axis('off')
            axs[i, 1].imshow(np.squeeze(attribution_pred_mask[i]), cmap=cmap, norm=colors.Normalize(vmax=vmax))
            axs[i, 1].axis('off')
            axs[i, 2].imshow(np.squeeze(attribution_pred_mask[i]), cmap=cmap, norm=colors.Normalize(vmax=vmax))
            axs[i, 2].imshow(image_show, alpha=overlay_alpha)
            axs[i, 2].axis('off')
            if clip:
                axs[i, 2].annotate('min: %.3f, max: %.3e' % (np.amin(attribution_pred_mask[i]), vmax), xy=(1, -0.06),
                                   xycoords='axes fraction', ha='right', va='bottom')
            else:
                axs[i, 2].annotate('min: %d, max: %d' % (np.amin(attribution_pred_mask[i]), vmax), xy=(1, -0.06),
                                   xycoords='axes fraction', ha='right', va='bottom')
    else:
        vmax = np.percentile(attribution_pred_mask, 99) if clip else 255
        axs[0, 1].imshow(np.squeeze(attribution_pred_mask), cmap=cmap, norm=colors.Normalize(vmax=vmax))
        axs[0, 1].axis('off')
        axs[0, 2].imshow(np.squeeze(attribution_pred_mask), cmap=cmap, norm=colors.Normalize(vmax=vmax))
        axs[0, 2].imshow(image_show, alpha=overlay_alpha)
        axs[0, 2].axis('off')
        if clip:
            axs[0, 2].annotate('min: %.3f, max: %.3e' % (np.amin(attribution_pred_mask), vmax), xy=(1, -0.06),
                               xycoords='axes fraction', ha='right', va='bottom')
        else:
            axs[0, 2].annotate('min: %d, max: %d' % (np.amin(attribution_pred_mask), vmax), xy=(1, -0.06),
                               xycoords='axes fraction', ha='right', va='bottom')

    # fig.tight_layout(h_pad=2)
    fig.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    # return fig


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

    print("Start: " + str(datetime.now()))

    if args.plot == 'cm':
        plot_confusion_matrix(args.data)
    elif args.plot == 'line':
        plot_sequence_classification_line_plot(args.data)
    elif args.plot == 'phase':
        phase_output_plot(args.data)
    elif args.plot == 'bbox':
        show_detection_bounding_box(data_path=args.data)

    print("\nEnd: " + str(datetime.now()))
