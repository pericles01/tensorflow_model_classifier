from src.amlw_deployment import AMLWDeployment
from src.experiment_config import ExperimentConfig
import argparse
import sys
import subprocess
import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

def run_script(cmd: list, wait: bool) -> None:
    logging.info("Run %s", ' '.join(cmd))
    script_process = subprocess.Popen(' '.join(cmd), stdout=subprocess.PIPE, shell=True)

    if wait:
        script_process.wait()

        if script_process.returncode != 0:
            exit(script_process.returncode)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the model with different complexities.')
    parser.add_argument('--target', type=str, help='name of compute instance to run the code')
    parser.add_argument('--config', default='configs/train__azure__optuna__model.yml', type=str, help="path to config file, see samples in folder configs/")
    args = parser.parse_args()

    setup_logger()

    logging.info('Using config ' + args.config)
    conf = ExperimentConfig(args.config)

    # if args.target == 'local-no-deploy':
    #     sys.path.append('./src')
    #     logging.info("Optuna local run")

    #     run_script(cmd=["python3", "train_optuna.py", "--config"], wait=True)
    
    # In case AMLW experiment tracking should not take place, detect args.target == 'local' and
    #   set some additional flag to force no tracking
    deployment = AMLWDeployment(args.target, conf)
    deployment.run('train_optuna.py')
