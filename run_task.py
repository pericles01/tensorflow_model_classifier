from src.amlw_deployment import AMLWDeployment
from src.experiment_config import ExperimentConfig
import argparse
import sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the model with different complexities.')
    parser.add_argument('--task', type=str, help='target task.task should be train, test or interpret')
    parser.add_argument('--target', type=str, help='name of compute instance to run the code')
    parser.add_argument('--config', type=str, help="path to config file, see samples in folder configs/")
    args = parser.parse_args()

    assert args.task in ["train", "test", "interpret"], "task should be 'train', 'test', or 'interpret' "
    print("Starting" + args.task + " task")
    print('Using config ' + args.config)
    conf = ExperimentConfig(args.config)

    # In case AMLW experiment tracking should not take place, detect args.target == 'local' and
    #   set some additional flag to force no tracking
    if args.target == 'local-no-deploy':
        sys.path.append('./src') # Needed to load more modules in train.py
        if args.task == "train":
            from src.train import train
            train(conf)
        elif args.task == "test":
            from src.test import test
            test(conf)
        else:
            from src.interpret import interpret
            interpret(conf)
    else:
        deployment = AMLWDeployment(args.target, conf)
        if args.task == "train":
            deployment.run('train.py')
        elif args.task == "test":
            deployment.run('test.py')
        else:
            deployment.run('interpret.py')