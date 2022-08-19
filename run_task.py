from src.amlw_deployment import AMLWDeployment
from src.experiment_config import ExperimentConfig
import argparse
import sys
#import logging


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the model with different complexities.')
    parser.add_argument('--task', type=str, help='target task.task should be train, test or interpret')
    parser.add_argument('--target', type=str, help='name of compute instance to run the code')
    parser.add_argument('--config', type=str, help="path to config file, see samples in folder configs/")
    args = parser.parse_args()

    # # Set up root logger, and add a file handler to root logger
    # logging.basicConfig(filename = 'stdout.log',
    #                 level = logging.DEBUG, 
    #                 filemode = "w",
    #                 format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')


    assert args.task in ["train", "test", "interpret"], "task should be 'train', 'test', or 'interpret' "
    # logging.info("Starting * " + args.task + " * task")
    # logging.info('Using config ' + args.config)
    print("Starting * " + args.task + " * task")
    print('Using config ' + args.config)
    conf = ExperimentConfig(args.config)

    # In case AMLW experiment tracking should not take place, detect args.target == 'local' and
    #   set some additional flag to force no tracking
    if args.target == 'local-no-deploy':
        sys.path.append('./src') # Needed to load more modules
        if args.task == "train":
            from src.train import train
            train(conf, local=True)
        elif args.task == "test":
            from src.test import test
            test(conf, local=True)
        else:
            from src.interpret import interpret
            interpret(conf, local=True)
    else:
        deployment = AMLWDeployment(args.target, conf)
        if args.task == "train":
            deployment.run('train.py')
        elif args.task == "test":
            deployment.run('test.py')
        else:
            deployment.run('interpret.py')