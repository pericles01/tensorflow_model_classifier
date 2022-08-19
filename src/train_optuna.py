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
from utils import misc, sched
from dataset import DataPaths, ClassificationDataset
import models
import argparse
from azureml.core import Run
from main_experiment import Exp
import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState
#import mlflow
#import mlflow.tensorflow
#from mlflow.tracking import get_tracking_uri


def train(trial) -> float:
    
    # Clear clutter from previous Keras session graphs.
    tf.keras.backend.clear_session()

    #azure_tags(args)

    # Define GPU devices and allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Start a new mlflow run
    #with mlflow.start_run(run_name="mlflow-optuna-test") as run:
    
    # Create dataset
    apply_tfms = trial.suggest_categorical("apply_tfms", [True, False])
    batch = trial.suggest_int("Batch size", low=8, high=64, step=8)
    train_data = ClassificationDataset(data_paths.train_img, data_paths.train_gt, args, optuna_tfms=apply_tfms, trial=trial, batch=batch)
    valid_data = ClassificationDataset(data_paths.valid_img, data_paths.valid_gt, args, split='validation', optuna_tfms=apply_tfms, trial=trial, batch=batch)

    # Create model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Classification(model_name=args.backbone, dims=args.dim,
                                            lr=trial.suggest_float("learning_rate", low=1e-3, high=1e-1, log=True), loss=args.loss, train_backbone=args.train_backbone,
                                            num_classes=args.num_classes, num_labels=args.num_labels,
                                            dropout=args.dropout, bayesian=args.bayesian)

    # Setup logging path and learning decay scheduler
    start_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = path.join(args.outdir, "logs", start_time_stamp)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0)
    lr_callback = sched.step_lr_decay(args=args)

    # # Enable auto-logging to MLflow to capture TensorBoard metrics.
    # mlflow.tensorflow.autolog()

    # Train
    history = model.model.fit(
                train_data.dataset,
                steps_per_epoch=train_samples // batch,
                validation_data=valid_data.dataset,
                validation_steps=valid_samples // batch,
                epochs=trial.suggest_int("Number of training epochs", low=10, high=50, step=5),
                callbacks=[tensorboard_callback, lr_callback]
            )
        
    history_log = history.history

        # #log to mlflow
        # mlflow.log_params(history_log)

        # # Get hyperparameter suggestions created by Optuna and log them as params using mlflow
        # mlflow.log_params(trial.params)

        # #log metrics to azurew
        # for keys in history_log:
        #     history_log[keys] = [float(x) for x in history_log[keys]]
        #     run.log_list(str(keys), history_log[keys])

    target_metrics = [float(x) for x in history_log["val_CustomMccMetric"]]
    target_metrics.sort()

        # Save model, training parameters, logs and weights
        # end_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # model_path = os.path.join(args.outdir, "models")
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path, exist_ok=True)

        # save_path = os.path.join(model_path, start_time_stamp)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path, exist_ok=True)
            
        # misc.create_model_fit_history_log_file(history.history,
        #                                            os.path.join(save_path, 'train_log.json'))
        
    #returns the max value of the list
    return target_metrics[-1]

    
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
    # if exp.epochs:
    #     run.tag('epochs', exp.epochs)
    # if exp.batch:
    #     run.tag('batch', exp.batch)
    if exp.validation_split:
        run.tag('validation_split', exp.validation_split)
    # if exp.lr:
    #     run.tag('lr', exp.lr)
    if exp.lr_decay:
        run.tag('lr_decay', exp.lr_decay)
    if exp.dropout:
        run.tag('dropout', exp.dropout)   
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
    args = Exp(conf)

    #fetch training and validation data paths
    data_paths = DataPaths(args=args)
    train_samples = len(data_paths.train_img)
    valid_samples = len(data_paths.valid_img)
    print("Training samples: " + str(train_samples))
    print("Validation samples: " + str(valid_samples))

    #mlflow tracking
    # if args.experiment:
    #     mlflow.set_experiment(args.experiment)
    # if args.azureml_mlflow_uri:
    #     mlflow.set_tracking_uri(args.azureml_mlflow_uri)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(train, n_trials=args.optuna_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # print("mlflow tracking URI: ", get_tracking_uri())

    
