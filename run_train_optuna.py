from src.amlw_deployment import AMLWDeployment
from src.experiment_config import ExperimentConfig
import argparse
import sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the model with different complexities.')
    parser.add_argument('--target', type=str, help='name of compute instance to run the code')
    parser.add_argument('--config', default='configs/train__azure__optuna__model.yml', type=str, help="path to config file, see samples in folder configs/")
    args = parser.parse_args()

    print('Using config ' + args.config)
    conf = ExperimentConfig(args.config)

    # In case AMLW experiment tracking should not take place, detect args.target == 'local' and
    #   set some additional flag to force no tracking
    deployment = AMLWDeployment(args.target, conf)
    deployment.run('train_optuna.py')
