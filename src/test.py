"""
Description
-----------
This script includes functions to perform evaluation and
prediction for the trained models
"""

import os
import ssl

# Logger Control 0 (all messages are logged) - 3 (no messages are logged)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Avoid SSL certification
ssl._create_default_https_context = ssl._create_unverified_context
#from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
from utils import data, misc, postproc, plot
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import models
import shutil, ntpath, csv, cv2, time
from datetime import datetime
from dataset import ClassificationDataset
import json, copy
from main_experiment import Exp
import argparse
from azureml.core import Run

THRESHOLD = .5

def test(conf, local=False):

    args = Exp(conf)
    if not local:
        azure_tags(args)

    if args.csv_eval:
        evaluate_demonstrator_predictions(args.indir, args.csv_eval, args)

    else:
        # Load model
        print("Loading model ...")
        classifier = models.Classification(model_name=args.backbone, dims=args.dim,
                                           lr=args.lr, loss=args.loss, train_backbone=args.train_backbone,
                                           num_classes=args.num_classes, num_labels=args.num_labels,
                                           dropout=args.dropout, bayesian=args.bayesian)
        try:
            classifier.model.load_weights(filepath=args.saved_model + "/weights.h5")
        except:  # Attempt to load older models
            classifier.model = tf.keras.models.load_model(args.saved_model)
        classifier.to_functional()
        print("Model loaded successfully")

        if args.class_eval:
            if args.num_labels:
                print("Evaluating multi-label classifier")
                evaluate_multi_label_classification(data_path=args.indir, model=classifier.model, args=args, local=local)
            elif args.num_classes == 2:
                print("Evaluating binary classifier")
                evaluate_binary_classification(data_path=args.indir, model=classifier.model, args=args, local=local)
            elif args.num_classes > 2:
                print("Evaluating multi-class classifier")
                evaluate_multi_class_classification(data_path=args.indir, model=classifier.model, args=args, local=local)

        elif args.class_sort:
            print("Sorting classification output")
            sort_classification_output(args.indir, classifier.model, args=args)

        elif args.class_table:
            print("Creating classification prediction table")
            create_classification_predictions_table(data_path=args.indir, model=classifier.model, args=args)

        elif args.class_overlay:
            print("Creating a prediction overlay on classification test set")
            classification_output_overlay(args.indir, classifier.model, args=args)

        else:
            raise Exception(
                "Please specify testing parameter. Available options --class_sort --class_table --class_eval"
                " --class_overlay Refer to README.md for more information")

    print("\nEnd: " + str(datetime.now()))
    print("Results saved in " + export_dir)

def evaluate_demonstrator_predictions(data_path, preds_path, args):
    """
    Calculate various prediction metrics over predictions for a demonstrator and plot the confusion matrix
    """
    frames, gts_all, class_names = data.get_classification_data_paths(data_path, args)
    _, preds_all = misc.get_demo_predictions(preds_path)

    if args.num_labels == 0:
        sorted_gts = zip(*sorted(zip(frames, gts_all)))
        frames, gts_all = [list(tuple) for tuple in sorted_gts]

    gts_all = np.asarray(gts_all)
    preds_all = np.asarray(preds_all)
    with open(export_dir + '/test_log.json', 'a') as f:
        json.dump({"Data path": args.indir, "Eval File": preds_path}, f, indent=2)
    # multi_label case
    if args.num_labels:
        preds_all[preds_all > THRESHOLD] = 1
        preds_all[preds_all <= THRESHOLD] = 0
        preds_all = preds_all.astype(int)

        calculate_multi_label_classification_metrics(args, preds_all, gts_all, export_dir)
        generate_multi_label_confusion_matrices(preds_all, gts_all)
    else:
        if args.num_classes == 2:
            preds_all = np.squeeze(preds_all)
            preds_all[preds_all > THRESHOLD] = 1
            preds_all[preds_all <= THRESHOLD] = 0
            preds_all = preds_all.astype(int)

            output_path = export_dir
            tp, tn, fp, fn = calculate_binary_metrics(preds_all, gts_all, output_path)
            plot_labels = args.class_names if args.class_names else np.array([0, 1])
            plot.confusion_matrix(cm_input=np.array([[tn, fp], [fn, tp]]), labels=plot_labels,
                                  savepath=output_path + "/Figure_1.png", title='Confusion matrix', normalize=True)

        else:  # TODO not tested
            preds_all = np.argmax(preds_all, axis=1)
            calculate_multi_class_metrics(preds_all, gts_all, export_dir)

def evaluate_binary_classification(data_path, model, args, local= False):
    """Calculate various prediction metrics over a test dataset for a binary classification model
    and plot the confusion matrix"""

    def _variation_ratio(bayesian_output):
        labels = np.round(bayesian_output).astype(int).squeeze()  # TODO apply threshold
        labels_normed = np.asarray([(np.bincount(labels[:, i], minlength=2) /
                                     len(bayesian_output)) for i in range(labels.shape[1])])
        return 0.5 - abs(labels_normed - 0.5)

    img_paths, gts_all, _ = data.get_classification_data_paths(data_path, args)
    assert gts_all, "Ground truth is required for model evaluation"
    data_loader = ClassificationDataset(img_paths, gts_all, args, split='test')

    raw_output = np.zeros((len(gts_all)))
    gts_all = np.asarray(gts_all)
    certainty_var = np.zeros(len(img_paths))
    certainty_meanvarratio = np.zeros(len(img_paths))
    certainty_maxvarratio = np.zeros(len(img_paths))
    if args.temp_filter:
        # initialize temporal filter object
        f = postproc.TemporalFilter(num_classes=args.num_classes, num_labels=args.num_labels,
                                    state_filter=args.state_filter, filter_size=args.filter_size, threshold=THRESHOLD)
        label_filtered = np.zeros_like(raw_output)
        raw_filtered = np.zeros_like(raw_output)
    i = 0
    start_timer = time.time()
    for img_batch, gt_batch in data_loader.dataset:
        if args.bayesian:
            bayesian_output = [model.predict(img_batch) for _ in range(args.bayesian_samples)]
            raw_batch = np.stack(bayesian_output).mean(axis=0)
            raw_output[i:i + len(raw_batch)] = np.round(raw_batch, 3).squeeze()

            var_batch_ratio = _variation_ratio(bayesian_output)
            certainty_var[i:i + len(raw_batch)] = np.var(bayesian_output, axis=0).squeeze()
            certainty_meanvarratio[i:i + len(raw_batch)] = np.mean(var_batch_ratio, axis=1).squeeze()
            certainty_maxvarratio[i:i + len(raw_batch)] = np.max(var_batch_ratio, axis=1).squeeze()

        else:
            raw_batch = model.predict(img_batch).squeeze()
            raw_output[i:i + len(raw_batch)] = np.round(raw_batch, 3)
        if args.temp_filter:
            for idx in range(len(img_batch)):
                label_filtered[i + idx], raw_filtered[i + idx] = f._filter_prediction(raw_batch[idx])

        i += len(raw_batch)
        print('\r Processed image %d / %d' % (i, len(img_paths)), end="")
    total_time = time.time() - start_timer
    print("\nMean FPS: " + str(np.round(1 / (total_time / len(img_paths)), 2)))
    if args.bayesian:
        print("Mean Certainty (Variance): ", str(np.round(np.mean(certainty_var), 3)))
        print("Mean Certainty (Mean(VarRatio)): ", str(np.round(np.mean(certainty_meanvarratio), 3)))
        print("Mean Certainty (Max(VarRatio)): ", str(np.round(np.mean(certainty_maxvarratio), 3)))
    labels = copy.deepcopy(raw_output)
    labels[labels > THRESHOLD] = 1
    labels[labels <= THRESHOLD] = 0

    output_path = export_dir

    with open(output_path + '/test_log.json', 'a') as f:
        json.dump({"Data path": args.indir, "Model path": args.saved_model}, f, indent=2)

    tp, tn, fp, fn = calculate_binary_metrics(labels, gts_all, output_path, local=local)
    plot_labels = args.class_names if args.class_names else np.array([0, 1])
    plot.confusion_matrix(cm_input=np.array([[tn, fp], [fn, tp]]), labels=plot_labels,
                          savepath=output_path + "/Figure_1.png", title='Confusion matrix', normalize=True, local=local)
    if args.temp_filter:
        tp, tn, fp, fn = calculate_binary_metrics(label_filtered, gts_all, output_path, text='binary_metrics_filtered', local=local)
        plot.confusion_matrix(cm_input=np.array([[tn, fp], [fn, tp]]), labels=plot_labels,
                              savepath=output_path + "/Figure_2.png", title='Confusion matrix filtered', normalize=True, local=local)
        plot.sequence_classification_line_plot_v2(np.array([gts_all, raw_output, label_filtered, raw_filtered]),
                                                  "Line plot", THRESHOLD)


def evaluate_multi_class_classification(data_path, model, args, local=False):
    """Calculate various prediction metrics over a test dataset for a multiclass classification model
    and plot the confusion matrix"""

    def _variation_ratio(bayesian_output):
        labels = np.argmax(bayesian_output, axis=2)
        labels_normed = np.asarray([(np.bincount(labels[:, i], minlength=bayesian_output[0].shape[1]) /
                                     len(bayesian_output)) for i in range(labels.shape[1])])
        return 0.5 - abs(labels_normed - 0.5)

    img_paths, gts_all, _ = data.get_classification_data_paths(data_path, args)
    assert gts_all, "Ground truth is required for model evaluation"
    data_loader = ClassificationDataset(img_paths, gts_all, args, split='test')

    labels = np.zeros((len(gts_all)))
    gts_all = np.asarray(gts_all)
    confusion_mat = np.zeros((args.num_classes, args.num_classes))
    certainty_meanvar = np.zeros(len(img_paths))
    certainty_maxvar = np.zeros(len(img_paths))
    certainty_meanvarratio = np.zeros(len(img_paths))
    certainty_maxvarratio = np.zeros(len(img_paths))
    if args.temp_filter:
        # initialize temporal filter object
        f = postproc.TemporalFilter(num_classes=args.num_classes, num_labels=args.num_labels,
                                    state_filter=args.state_filter, filter_size=args.filter_size, threshold=THRESHOLD)
        label_filtered = np.zeros_like(labels)
    i = 0
    start_timer = time.time()
    for img_batch, gt_batch in data_loader.dataset:
        if args.bayesian:
            bayesian_output = [model.predict(img_batch) for _ in range(args.bayesian_samples)]
            raw_batch = np.stack(bayesian_output).mean(axis=0)
            labels[i:i + len(raw_batch)] = np.argmax(raw_batch, axis=1)

            var_batch = np.var(bayesian_output, axis=0)
            var_batch_ratio = _variation_ratio(bayesian_output)
            certainty_meanvar[i:i + len(raw_batch)] = np.mean(var_batch, axis=1)
            certainty_maxvar[i:i + len(raw_batch)] = np.max(var_batch, axis=1)
            certainty_meanvarratio[i:i + len(raw_batch)] = np.mean(var_batch_ratio, axis=1)
            certainty_maxvarratio[i:i + len(raw_batch)] = np.max(var_batch_ratio, axis=1)
        else:
            raw_batch = model.predict(img_batch)
            labels[i:i + len(raw_batch)] = np.argmax(raw_batch, axis=1)
        if args.temp_filter:
            for idx in range(len(img_batch)):
                label_filtered[i + idx], _ = f._filter_prediction(np.round(raw_batch[idx], 3))

        i += len(raw_batch)
        print('\r Processed image %d / %d' % (i, len(img_paths)), end="")
    total_time = time.time() - start_timer
    print("\nMean FPS: " + str(np.round(1 / (total_time / len(img_paths)), 2)))
    if args.bayesian:
        res = {
            "Mean Certainty (Mean(Variance))": float(np.round(np.mean(certainty_meanvar), 3)),
            "Mean Certainty (Max(Variance))": float(np.round(np.mean(certainty_maxvar), 3)),
            "Mean Certainty (Mean(VarRatio))": float(np.round(np.mean(certainty_meanvarratio), 3)),
            "Mean Certainty (Max(VarRatio))": float(np.round(np.mean(certainty_maxvarratio), 3))
        }
        print("/n", res)

    #output_path = export_dir
    with open(export_dir + '/test_log.json', 'a') as f:
        json.dump({"Data path": args.indir, "Model path": args.saved_model}, f, indent=2)

    calculate_multi_class_metrics(labels, gts_all, export_dir)
    for j in range(args.num_classes):
        for k in range(args.num_classes):
            confusion_mat[j, k] = np.sum(np.logical_and(gts_all == j, labels == k))
    plot_labels = args.class_names if args.class_names else np.arange(args.num_classes)
    output_path = os.path.join(export_dir, "Figure_1.png")
    plot.confusion_matrix(cm_input=confusion_mat, labels=plot_labels, savepath=output_path, normalize=True, local=local)

    if args.temp_filter:
        calculate_multi_class_metrics(labels, gts_all, export_dir,
                                      'Multi-class metrics filtered')
        for j in range(args.num_classes):
            for k in range(args.num_classes):
                confusion_mat[j, k] = np.sum(np.logical_and(gts_all == j, label_filtered == k))
        print("Mean accuracy after filtering: " + str(np.round(np.mean(label_filtered == gts_all), 3)))
        output_path = os.path.join(export_dir, "Figure_2.png")
        plot.confusion_matrix(cm_input=confusion_mat, labels=plot_labels, savepath=output_path,
                              title="Confusion matrix filtered", normalize=True, local=local)
        plot.sequence_classification_line_plot(np.array([gts_all, labels, label_filtered]), args.num_classes,
                                               title="Line plot",
                                               labels=["Ground Truth", "Raw output", "Label filtered"],
                                               subplot=False)


def evaluate_multi_label_classification(data_path, model, args, local=False):
    """Calculate various prediction metrics over a test dataset for a multi-label classification model
    and plot the confusion matrix"""

    def _variation_ratio(bayesian_output):
        labels = np.asarray(bayesian_output)
        labels = np.asarray([np.where(labels[:, i, :] >= THRESHOLD)
                             for i in range(labels.shape[1])])[:, 1]
        labels_normed = np.asarray([(np.bincount(labels[i], minlength=bayesian_output[0].shape[1]) /
                                     len(bayesian_output)) for i in range(len(labels))])
        return 0.5 - abs(labels_normed - 0.5)

    num_labels = args.num_labels
    img_paths, gts_all, _ = data.get_classification_data_paths(data_path, args)

    data_loader = ClassificationDataset(img_paths, gts_all, args,  split='test')

    assert gts_all, "Ground truth is required for model evaluation"
    gts_all = np.asarray(gts_all)
    raw_output = np.zeros((len(img_paths), num_labels), dtype=int)
    certainty_meanvar = np.zeros(len(img_paths))
    certainty_maxvar = np.zeros(len(img_paths))
    certainty_meanvarratio = np.zeros(len(img_paths))
    certainty_maxvarratio = np.zeros(len(img_paths))
    if args.temp_filter:
        # initialize temporal filter object
        f = postproc.TemporalFilter(num_classes=args.num_classes, num_labels=args.num_labels,
                                    state_filter=args.state_filter, filter_size=args.filter_size, threshold=THRESHOLD)
        labels_filtered = np.zeros_like(raw_output)
    i = 0
    start_timer = time.time()
    for img_batch, gt_batch in data_loader.dataset:
        if args.bayesian:
            bayesian_output = [model.predict(img_batch) for _ in range(args.bayesian_samples)]
            raw_batch = np.stack(bayesian_output).mean(axis=0)
            raw_output[i:i + len(raw_batch)] = np.round(raw_batch, 3)

            var_batch = np.var(bayesian_output, axis=0)
            var_batch_ratio = _variation_ratio(bayesian_output)
            certainty_meanvar[i:i + len(raw_batch)] = np.mean(var_batch, axis=1)
            certainty_maxvar[i:i + len(raw_batch)] = np.max(var_batch, axis=1)
            certainty_meanvarratio[i:i + len(raw_batch)] = np.mean(var_batch_ratio, axis=1)
            certainty_maxvarratio[i:i + len(raw_batch)] = np.max(var_batch_ratio, axis=1)

        else:
            raw_batch = model.predict(img_batch)
            raw_output[i:i + len(raw_batch)] = np.round(raw_batch, 3)
        if args.temp_filter:
            for idx in range(len(img_batch)):
                # In: [1.0, 0.0, 0.0, 0.027]
                labels_filtered[i + idx], _ = f._filter_prediction(np.round(raw_batch[idx], 3))
                # Out: ([1,0,0,0],[1.0,0.1,0.0,0.2])
        i += len(raw_batch)
        print('\r Processing image %d / %d' % (i, len(img_paths)), end="")
    total_time = time.time() - start_timer
    print("\nMean FPS: " + str(np.round(1 / (total_time / len(img_paths)), 2)))
    if args.bayesian:
        res = {
            "Mean Certainty (Mean(Variance))": float(np.round(np.mean(certainty_meanvar), 3)),
            "Mean Certainty (Max(Variance))": float(np.round(np.mean(certainty_maxvar), 3)),
            "Mean Certainty (Mean(VarRatio))": float(np.round(np.mean(certainty_meanvarratio), 3)),
            "Mean Certainty (Max(VarRatio))": float(np.round(np.mean(certainty_maxvarratio), 3))
        }
        print(res)
        if not local:
            #log into Azure
            for dict in res:
                run.log(str(dict), res[dict] )

    with open(export_dir + '/test_log.json', 'a') as f:
        json.dump({"Data path": args.indir, "Model path": args.saved_model}, f, indent=2)

    labels = copy.deepcopy(raw_output)
    labels[labels > THRESHOLD] = 1
    labels[labels <= THRESHOLD] = 0

    calculate_multi_label_classification_metrics(labels, gts_all, export_dir)
    generate_multi_label_confusion_matrices(labels, gts_all)

    if args.temp_filter:
        calculate_multi_label_classification_metrics(labels_filtered, gts_all,
                                                     export_dir,
                                                     "Multi-label metrics filtered", filtered=True)
        generate_multi_label_confusion_matrices(labels_filtered, gts_all, "_filtered")


def calculate_binary_metrics(preds, gts, output_path, text="binary_metrics", local=False):
    """Return binary classification metrics, given prediction and ground truth"""
    acc = np.mean(preds == gts)
    tp = np.sum(np.logical_and(preds == 1, gts == 1))
    tn = np.sum(np.logical_and(preds == 0, gts == 0))
    fp = np.sum(np.logical_and(preds == 1, gts == 0))
    fn = np.sum(np.logical_and(preds == 0, gts == 1))
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    #Calculate MCC
        # MCC = (TP * TN) - (FP * FN) /
        #   ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)
    mcc = ((tp*tn)-(fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    res = {text:
               {"Acc": float(np.round(acc, 3)),
                "TP": int(tp), "TN": int(tn),
                "FP": int(fp), "FN": int(fn),
                "Sensitivity": float(np.round(sens, 3)),
                "Specificity": float(np.round(spec, 3)),
                "Precision": float(np.round(prec, 3)),
                "F1_micro": float(np.round(f1, 3)),
                "Balanced accuracy": float(np.round((sens + spec) / 2, 3)),
                "MatthewsCorrelationCoefficient": float(np.round(mcc, 3))}
           }
    print("\n", res)
    if not local:
        #log into Azure
        for dict in res[text]:
            run.log(str(dict), res[text][dict] )

    with open(output_path + '/test_log.json', 'a') as f:
        json.dump(res, f, indent=2)
    return tp, tn, fp, fn


def calculate_multi_class_metrics(predicted_labels, gts, output_path, text='Multi-class metrics'):
    balanced_acc = []
    for i in range(args.num_classes):
        balanced_acc.append(balanced_accuracy_score(gts == i, predicted_labels == i))
    res = {text: {
        "Mean accuracy": np.round(np.mean(predicted_labels == gts), 3),
        "Micro F1-Score": np.round(f1_score(gts, predicted_labels, average='micro'), 3),
        "Macro F1-Score": np.round(f1_score(gts, predicted_labels, average='macro'), 3),
        "Weighted F1-Score": np.round(f1_score(gts, predicted_labels, average='weighted'), 3),
        "Per-class F1-Scores": list(np.round(f1_score(gts, predicted_labels, average=None), 3), ),
        "Per-class balanced accuracy": list(np.round(balanced_acc, 3))
    }}
    print("\n", res)
    #log into Azure
    for dict in res[text]:
        run.log(str(dict), res[text][dict] )

    with open(output_path + '/test_log.json', 'a') as f:
        json.dump(res, f, indent=2)


def calculate_multi_label_classification_metrics(predicted_labels, gts, output_path, text="Multi-label metrics",
                                                 filtered=False, local=False):
    calculate_binary_metrics(predicted_labels, gts, export_dir,
                             "Binary metrics filtered" if filtered else "Binary metrics", local=local)
    for i in range(args.num_labels):
        calculate_binary_metrics(predicted_labels[:, i], gts[:, i], export_dir,
                                 "Binary metrics filtered, label %s" % i if filtered
                                 else "Binary metrics, label %s" % i, local=local)
    balanced_acc = []
    for i in range(gts.shape[1]):
        balanced_acc.append(balanced_accuracy_score(gts[:, i], predicted_labels[:, i]))

    res = {text: {
        "Label-wise accuracies": list(np.round(np.mean(predicted_labels == gts, axis=0), 3)),
        "Sample-wise accuracy": np.round(accuracy_score(predicted_labels, gts), 3),
        "Micro F1-Score": np.round(f1_score(gts, predicted_labels, average='micro'), 3),
        "Macro F1-Score": np.round(f1_score(gts, predicted_labels, average='macro'), 3),
        "Weighted F1-Score": np.round(f1_score(gts, predicted_labels, average='weighted'), 3),
        "Per-sample F1-Score": np.round(f1_score(gts, predicted_labels, average='samples', zero_division=0), 3),
        "Per-label F1-Scores": list(np.round(f1_score(gts, predicted_labels, average=None), 3)),
        "Per-label balanced accuracy": list(np.round(balanced_acc, 3)),
    }}
    print("\n", res)
    if not local:
        #log into Azure
        for dict in res[text]:
            run.log(str(dict), res[text][dict] )
        
    with open(output_path + '/test_log.json', 'a') as f:
        json.dump(res, f, indent=2)


def generate_multi_label_confusion_matrices(predicted_labels, gts, filtered=""):
    def mask(num_labels, true_vals):
        m = np.zeros(num_labels, dtype=int)
        for elem in true_vals:
            m[elem] = 1
        return m

    matrix = np.zeros((args.num_labels, args.num_labels))
    labels_names = args.class_names if args.class_names else list(np.arange(args.num_labels))
    for i in range(args.num_labels):
        tp = np.sum(np.logical_and(predicted_labels[:, i] == 1, gts[:, i] == 1))
        tn = np.sum(np.logical_and(predicted_labels[:, i] == 0, gts[:, i] == 0))
        fp = np.sum(np.logical_and(predicted_labels[:, i] == 1, gts[:, i] == 0))
        fn = np.sum(np.logical_and(predicted_labels[:, i] == 0, gts[:, i] == 1))
        output_path = os.path.join(export_dir, "Figure_%d%s.png" % (i, filtered))
        plot.confusion_matrix(cm_input=np.array([[tn, fp], [fn, tp]]), labels=np.array([0, 1]), savepath=output_path,
                              title='Label: ' + str(labels_names[i]) + filtered, normalize=True)

    for gt_idx in range(args.num_labels):
        for pred_idx in range(args.num_labels):
            matrix[gt_idx, pred_idx] = np.sum(np.logical_and(
                (np.bitwise_and(gts, mask(args.num_labels, [gt_idx, pred_idx])) == mask(args.num_labels,
                                                                                        [gt_idx])).all(axis=1),
                (np.bitwise_and(predicted_labels, mask(args.num_labels, [gt_idx, pred_idx])) == mask(args.num_labels,
                                                                                                     [pred_idx])).all(
                    axis=1)))

    plot_labels = args.class_names if args.class_names else np.arange(args.num_labels)
    cm_input = matrix.astype('float') / gts.sum(axis=0)[:, np.newaxis]
    cm_input = np.nan_to_num(cm_input)
    plot.confusion_matrix(cm_input=cm_input, labels=plot_labels, normalize=False,
                          savepath=os.path.join(export_dir,
                                                "Figure_%d%s.png" % (args.num_labels, filtered)),
                          title='Multi-Label Confusion Matrix' + filtered, cmap='Purples')
    plot.confusion_matrix(cm_input=matrix, labels=plot_labels, normalize=False,
                          savepath=os.path.join(export_dir,
                                                "Figure_%d%s.png" % (args.num_labels + 1, filtered)),
                          title='Multi-Label Confusion Matrix (absolute)' + filtered, cmap='Purples')


def classification_output_overlay(data_path, model, args):
    """Overlay the classification (binary or multi-class) raw output, predicted label and the ground truth (if given)
    over test images. Press any key to go to next image. Press q to exit """

    img_paths, gts_all, class_names = data.get_classification_data_paths(data_path, args)
    if gts_all:
        gts_flag = True
    else:
        gts_all = [0] * len(img_paths)
        gts_flag = False
    data_loader = ClassificationDataset(img_paths, gts_all, args, split='test')
    if args.temp_filter:  # TODO adapt to batch inference
        # initialize temporal filter object
        f = postproc.TemporalFilter(num_classes=args.num_classes, num_labels=args.num_labels,
                                    state_filter=args.state_filter, filter_size=args.filter_size, threshold=THRESHOLD)
    i = 0
    for img_batch, gt_batch in data_loader.dataset:
        # print("Processing image" + str(i + 1) + " / " + str(len(img_paths)) + "     "
        # "Buffer:" + str(np.round(np.ndarray.tolist(np.asarray(pred_buffer)),3))) # Activate for debugging
        file_names = [ntpath.basename(img_paths[idx]) for idx in range(i, i + len(img_batch))]
        imgs_orig = [cv2.imread(img_paths[idx]) for idx in range(i, i + len(img_batch))]
        raw_batch = np.round(model.predict(img_batch), 3)
        gts_batch = gts_all[i:i + len(img_batch)] if gts_flag else []
        if args.temp_filter:
            label_filtered = [0] * len(raw_batch)
            for idx in range(len(img_batch)):
                label_filtered[idx], _ = f._filter_prediction(raw_batch[idx])

        for idx in range(len(img_batch)):
            img_out = misc.overlay_text(imgs_orig[idx], raw_batch[idx], args.num_classes, args.num_labels,
                                        label_filtered[idx] if args.temp_filter else None,
                                        gts_batch[idx] if gts_batch else [])
            cv2.imwrite(export_dir + '/' + file_names[idx], img_out)
        i += len(img_batch)
        print('\r Processed image %d / %d' % (i, len(img_paths)), end="")


def sort_classification_output(data_path, model, args):
    """ Sort images based on binary/multi-class classifier predictions
    works both for data samples with given/un-given gts"""
    img_paths, gts_all, class_names = data.get_classification_data_paths(data_path, args)
    if not gts_all:
        gts_all = [0] * len(img_paths)
    data_loader = ClassificationDataset(img_paths, gts_all, args, split='test')
    for i in range(len(class_names)):
        os.mkdir(export_dir + '/' + class_names[i])
    if args.temp_filter:
        f = postproc.TemporalFilter(num_classes=args.num_classes, num_labels=args.num_labels,
                                    state_filter=args.state_filter, filter_size=args.filter_size, threshold=THRESHOLD)
    i = 0
    for img_batch, gt_batch in data_loader.dataset:
        file_names = [ntpath.basename(img_paths[idx]) for idx in range(i, i + len(img_batch))]
        raw_batch = np.round(model.predict(img_batch), 3)
        if args.num_classes == 2:
            labels = [1 if val > THRESHOLD else 0 for val in raw_batch]
        else:
            labels = [np.argmax(val) for val in raw_batch]
        if args.temp_filter:
            for idx in range(len(img_batch)):
                labels[idx], _ = f._filter_prediction(raw_batch[idx])

        for idx in range(len(img_batch)):
            shutil.copy(img_paths[i + idx], os.path.join(export_dir, class_names[int(labels[idx])], file_names[idx]))

        i += len(img_batch)
        print('\r Processed image %d / %d' % (i, len(img_paths)), end="")


def create_classification_predictions_table(data_path, model, args):
    """Create a csv table containing image names, ground truths (if given), predicted classes and probabilities for both
     binary and multi-class classification models"""
    img_paths, gts_all, class_names = data.get_classification_data_paths(data_path, args)
    if gts_all:
        gts_flag = True
        header = ["Name", "Ground Truth", "Label", "Raw Output"]
    else:
        gts_all = [0] * len(img_paths)
        gts_flag = False
        header = ["Name", "Label", "Raw Output"]
    data_loader = ClassificationDataset(img_paths, gts_all, args, split='test')
    file_names = [ntpath.basename(img_paths[idx]) for idx in range(len(img_paths))]

    with open(export_dir + '/' + 'prediction.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        if args.temp_filter:  # TODO adapt to batch inference
            f = postproc.TemporalFilter(num_classes=args.num_classes, num_labels=args.num_labels,
                                        state_filter=args.state_filter, filter_size=args.filter_size,
                                        threshold=THRESHOLD)
            header.extend(['Label Filtered', 'Raw Filtered'])
        csv_writer.writerow(header)
        i = 0
        for img_batch, gt_batch in data_loader.dataset:
            if args.num_labels:
                raw_batch = model.predict(img_batch)
                label = [1 if val > THRESHOLD else 0 for raw_sample in raw_batch for val in raw_sample]

            elif args.num_classes == 2:
                raw_batch = model.predict(img_batch).squeeze()
                label = [1 if val > THRESHOLD else 0 for val in raw_batch]
                raw_batch = np.round(raw_batch, 3)

            elif args.num_classes > 2:
                raw_batch = model.predict(img_batch)
                raw_batch = np.round(raw_batch, 3)
                label = np.argmax(raw_batch, axis=1)
            for idx in range(len(img_batch)):
                if args.num_labels:
                    if gts_flag:
                        line = [str(file_names[i + idx]), str(gts_all[i + idx])[1:-1],
                                str((raw_batch[idx] > THRESHOLD).astype(int))[1:-1],
                                str(np.round(raw_batch[idx], 3))[1:-1]]
                    else:
                        line = [str(file_names[i + idx]), str((raw_batch[idx][0] > THRESHOLD).astype(int))[1:-1],
                                str(np.round(raw_batch[idx][0], 3))[1:-1]]
                else:
                    if gts_flag:
                        line = [file_names[i + idx], gts_all[i + idx], label[idx],
                                str(np.round(raw_batch[idx], 3))[1:-1]]
                    else:
                        if args.num_classes == 2:
                            line = [file_names[i + idx], label[idx], str(np.round(raw_batch[idx], 3))]
                        else:
                            line = [file_names[i + idx], label[idx], str(np.round(raw_batch[idx], 3))[1:-1]]

                if args.temp_filter:
                    label_filtered, raw_filtered = f._filter_prediction(np.round(raw_batch[idx], 3))
                    if args.num_labels:
                        line.extend([str((np.asarray(label_filtered) > THRESHOLD).astype(int))[1:-1],
                                     str(np.round(raw_filtered, 3))[1:-1]])
                    else:
                        line.extend([str(label_filtered)])
                        line.extend([str(raw_filtered)[1:-1]])

                csv_writer.writerow(line)

            i += len(img_batch)
            print('\r Processed image %d / %d' % (i, len(img_paths)), end="")
        csv_file.close()

def azure_tags(args):

    # ------------------------------------- Base arguments ------------------------------------- #
    run.tag('task', "classification test")
    if args.dim:
         run.tag('dim', args.dim)
    if args.resize:
        run.tag('resize', args.resize)
    if args.check_data:
        run.tag('check_data', args.check_data)
    if args.class_names:
        run.tag('class_names', args.class_names)
    # ------------------------------------- Model arguments ------------------------------------- #
    if args.backbone:
        run.tag('backbone', args.backbone)
    if args.num_classes:
        run.tag('num_classes', args.num_classes)
    if args.num_labels:
        run.tag('num_labels', args.num_labels)
    # ------------------------------------- Testing arguments ------------------------------------- #
    if args.bayesian:
        run.tag('bayesian', args.bayesian)
    if args.bayesian_samples:
        run.tag('bayesian_samples', args.bayesian_samples)
    if args.csv_eval:
        run.tag('csv_eval', args.csv_eval)
    if args.class_eval:
        run.tag('class_eval', args.class_eval)
    if args.class_overlay:
        run.tag('class_overlay', args.class_overlay)
    if args.class_table:
        run.tag('class_table', args.class_table)
    if args.class_sort:
        run.tag('class_sort', args.class_sort)
    if args.temp_filter:
        run.tag('temp_filter', args.temp_filter)
    if args.filter_size:
        run.tag('filter_size', args.filter_size)
    if args.state_filter:
        run.tag('state_filter', args.state_filter)


if __name__ == '__main__':
    # Need only these to be parsed because they are dynamically created by azureml SDK
    # All options are defined in the yaml file passed with --config
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--saved_model', type=str, help="saved model directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()
    
    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)

    result_path = os.path.join(args.outdir, "results")
    if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)

    timeStamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    export_dir = os.path.join(result_path, timeStamp)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    print("Start: " + str(datetime.now()))

    run = Run.get_context()
    test(conf)

    
