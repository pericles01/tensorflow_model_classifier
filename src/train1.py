"""Description
   -----------
This script is used to train classification and segmentation models
 based on MobileNet, Resnet and DenseNet architectures and log metrics using mlflow


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
from main_experiment import Exp
import mlflow
import mlflow.tensorflow
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


def train(conf):

    args = Exp(conf)

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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0)
    lr_callback = sched.step_lr_decay(args=args)
    with mlflow.start_run():
        mlflow.tensorflow.autolog()
        # Train
        history = model.model.fit(
                train_data.dataset,
                steps_per_epoch=train_samples // args.batch,
                validation_data=valid_data.dataset,
                validation_steps=valid_samples // args.batch,
                epochs=args.epochs,
                callbacks=[tensorboard_callback, lr_callback]
            )
         
        # history_log = history.history
        # for keys in history_log:
        #     history_log[keys] = [float(x) for x in history_log[keys]]
        #     #log metrics to mlflow
        #     mlflow.log_params(history_log)
          
    # Save model, training parameters, logs and weights
    end_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_path = os.path.join(args.outdir, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    save_path = os.path.join(model_path, start_time_stamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    misc.create_model_fit_history_log_file(history.history,
                                               os.path.join(save_path, 'train_log.json'))
    misc.create_config_file(args, save_path, validation_set=data_paths.valid_img,
                                aug_transforms=list(augment.generate_augmentation(args)), num_train=train_samples,
                                num_valid=valid_samples, start_time_stamp=start_time_stamp,
                                end_time_stamp=end_time_stamp)
    tf.saved_model.save(model.model, save_path)
    model.model.save_weights(save_path + "/weights.h5")

    logging.info("Model saved in " + save_path)

if __name__ == '__main__':
    # Need only these to be parsed because they are dynamically created by azureml SDK
    # All options are defined in the yaml file passed with --config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="path to config file of the experiment")
    args = parser.parse_args()
    setup_logger()

    logging.info('Using config ' + args.config)

    mlflow.set_experiment("mlflow-test-1")
    
    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)
    train(conf)

    
