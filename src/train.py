"""Description
   -----------
This script is used to train classification and segmentation models
 based on MobileNet, Resnet and DenseNet architectures


Parameters
----------
The parameters are fetched using an argparse and ExperimentConfig class, which is defined in experiment_config.py


Returns
-------
- trained model : .pb
the  trained model is saved in args.outdir/models/<timestamp>/
- model weights : .h5
the weights of the model are saved separately in  args.outdir/models/<timestamp>/
in case models were to be built again after training
- log file : tensorboard
log files are saved in args.outdir/logs/<timestamp>/
"""
import os
import ssl

# Logger Control 0 (all messages are logged) - 3 (no messages are logged)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Avoid SSL certification
ssl._create_default_https_context = ssl._create_unverified_context

from datetime import datetime
from os import path
import shutil, time
import numpy as np
import tensorflow as tf
from utils import misc, sched, augment
from dataset import DataPaths, ClassificationDataset #, transforms
import models
import argparse
from azureml.core import Run
from main_experiment import Exp


def train(conf, local=False):

    args = Exp(conf)
    
    #local train
    if not local:
        azure_tags(args)

    # Define GPU devices and allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Create dataset
    data_paths = DataPaths(args=args)
    train_data = ClassificationDataset(data_paths.train_img, data_paths.train_gt, args)
    valid_data = ClassificationDataset(data_paths.valid_img, data_paths.valid_gt, args, split='validation')

    train_samples = len(data_paths.train_img)
    valid_samples = len(data_paths.valid_img)
    print("Training samples: " + str(train_samples))
    print("Validation samples: " + str(valid_samples))

    # Create model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Classification(model_name=args.backbone, dims=args.dim,
                                        lr=args.lr, loss=args.loss, train_backbone=args.train_backbone,
                                        num_classes=args.num_classes, num_labels=args.num_labels,
                                        dropout=args.dropout, bayesian=args.bayesian)

    # Setup logging path and learning decay scheduler
    start_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = path.join(args.outdir, "logs", start_time_stamp)
    model_path = os.path.join(args.outdir, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    save_path = os.path.join(model_path, start_time_stamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    #callbacks    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0)
    lr_callback = sched.step_lr_decay(args=args)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_MCC', patience=5, mode='max') #,restore_best_weights=True #helpful?
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path+"/weights.h5", monitor='val_Custom_Mcc', save_best_only=True, mode='max',
                                                        save_weights_only=True)
    # Train
    history = model.model.fit(
            train_data.dataset,
            steps_per_epoch=train_samples // args.batch,
            validation_data=valid_data.dataset,
            validation_steps=valid_samples // args.batch,
            epochs=args.epochs,
            callbacks=[tensorboard_callback, lr_callback, checkpoint]
        )
    if not local:
        #log metrics to azure
        history_log = history.history
        for keys in history_log:
            history_log[keys] = [float(x) for x in history_log[keys]]
            run.log_list(str(keys), history_log[keys])
        
    # Save model, training parameters, logs and weights
    end_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    
        
    misc.create_model_fit_history_log_file(history.history,
                                               os.path.join(save_path, 'train_log.json'))
    misc.create_config_file(args, save_path, validation_set=data_paths.valid_img,
                                aug_transforms=list(augment.generate_augmentation(args)), num_train=train_samples,
                                num_valid=valid_samples, start_time_stamp=start_time_stamp,
                                end_time_stamp=end_time_stamp)
    tf.saved_model.save(model.model, save_path)
    #model.model.save_weights(save_path + "/weights.h5") #done already in checkpoint callback

    print("Model saved in " + save_path)

    
def azure_tags(exp):

    # ------------------------------------- Base arguments ------------------------------------- #

    run.tag('task', "classification train")
    if exp.dim:
         run.tag('dim', exp.dim)
    if exp.resize:
        run.tag('resize', exp.resize)
    if exp.check_data:
        run.tag('check_data', exp.check_data)
    if exp.class_names:
        run.tag('class_names', exp.class_names)
    if exp.commit_hash:
        run.tag('commit_hash', exp.commit_hash)
    # ------------------------------------- Model arguments ------------------------------------- #
    if exp.backbone:
        run.tag('backbone', exp.backbone)
    if exp.head:
        run.tag('head', exp.head)
    if exp.pooling:
        run.tag('pooling', exp.pooling)
    if exp.num_classes:
        run.tag('num_classes', exp.num_classes)
    if exp.num_labels:
        run.tag('num_labels', exp.num_labels)
    # ------------------------------------- Training arguments ------------------------------------- #
    if exp.epochs:
        run.tag('epochs', exp.epochs)
    if exp.batch:
        run.tag('batch', exp.batch)
    if exp.validation_split:
        run.tag('validation_split', exp.validation_split)
    if exp.lr:
        run.tag('lr', exp.lr)
    if exp.lr_decay:
        run.tag('lr_decay', exp.lr_decay)
    if exp.dropout:
        run.tag('dropout', exp.dropout)
    if len(exp.augment) > 0:
        run.tag('augmentation', exp.augment)
    else:
        run.tag('augmentation', "None")
    if exp.label_smoothing:
        run.tag('label_smoothing', exp.label_smoothing)
    if exp.train_backbone:
        run.tag('train_backbone', exp.train_backbone)
    if exp.kmeans:
        run.tag('kmeans', exp.kmeans)
    if exp.bayesian:
        run.tag('bayesian', exp.bayesian)
    if exp.bayesian_samples:
        run.tag('bayesian_samples', exp.bayesian_samples)

if __name__ == '__main__':
    # Need only these to be parsed because they are dynamically created by azureml SDK
    # All options are defined in the yaml file passed with --config
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--valdata', type=str, help="input validation data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--saved_model', type=str, help="saved model directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()

    run = Run.get_context()
    
    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)
    train(conf)

    
