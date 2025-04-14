#!/usr/bin/env python
import datetime
import itertools
import pandas as pd
import os
import sys
import time
from pathlib import Path
from pprint import pprint

# Paths
PROJECT_DIR          = ""  # i.e /home/vmasrani/dev/aquaeye-ml
EXPERIMENT_DIR       = ""  # i.e /home/vmasrani/dev/aquaeye-ml/experiments
NAMED_EXPERIMENT_DIR = ""  # i.e /home/vmasrani/dev/aquaeye-ml/experiments/first_run
RESULTS_DIR          = ""  # i.e results/{experiment_name}/{now.strftime("%Y_%m_%d_%H:%M:%S")}'
LOCAL_DIR   = ""  # i.e home/vaden/dev/projects/
SCRIPT_NAME = ""  # i.e main.py

SLEEP_TIME = 1.05
NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

PYTHON_PATH = '/opt/conda/envs/aquaeye-env/bin/python'

########################
# Main submission loop #
########################

def submit(hyper_params,
           experiment_dir,
           script_name,
           ):

    # Validate arguments
    experiment_name = Path(experiment_dir).parts[-1]
    verify_dirs(experiment_dir, experiment_name, script_name)

    if isinstance(hyper_params, dict):
        hyper_params['tags'] = [experiment_name] # add experiment name to tags
    else:
        for h in hyper_params:
            h['tags'] = [experiment_name] # add experiment name to tags

    hypers = process_hyperparameters(hyper_params)

    # Display info
    print("------Sweeping over------")
    pprint(hyper_params)
    print(f"-------({len(hypers)} runs)-------")

    ask = True
    flag = 'y'

    for idx, hyper in enumerate(hypers):
        if ask:
            flag = input(
                f"Submit ({idx + 1}/{len(hypers)}): {hyper}? (y/n/all/exit) "
            )
        if flag in ['yes', 'all', 'y', 'a']:
            job_name = f"{experiment_name}_{NOW}_{idx}"
            submit_job(hyper, job_name)
            print(f"({job_name}): {hyper}")

        if flag in ['all', 'a']:
            ask = False
            time.sleep(SLEEP_TIME)

        if flag in ['exit', 'e']:
            sys.exit()


##############################
# ---- path management -------
##############################

# Strictly enforce directory structure
# pylint: disable=global-statement
def verify_dirs(experiment_dir, experiment_name, script_name) -> None:
    # make global
    global NAMED_EXPERIMENT_DIR
    global EXPERIMENT_DIR
    global PROJECT_DIR
    global RESULTS_DIR
    global LOCAL_DIR
    global SCRIPT_NAME

    NAMED_EXPERIMENT_DIR = experiment_dir
    EXPERIMENT_DIR = Path(experiment_dir).parents[0]
    PROJECT_DIR = Path(experiment_dir).parents[1]
    LOCAL_DIR = Path(experiment_dir).parents[2]
    RESULTS_DIR = f'results/{experiment_name}/{NOW}'
    SCRIPT_NAME = script_name

    assert PROJECT_DIR.is_dir(), f"{PROJECT_DIR} does not exist"

#################################
# ------- hyperparameters -------
#################################


def make_hyper_string_from_dict(hyper_dict):

    # Check all values are iterable lists
    def type_check(value):
        return list(value) if isinstance(value, (list, range)) else [value]

    hyper_dict = {key: type_check(value) for key, value in hyper_dict.items()}
    commands = []
    for args in itertools.product(*hyper_dict.values()):
        header = list(zip(hyper_dict.keys(), args))
        commands.append(dict(header))
    return commands


def process_hyperparameters(hyper_params):
    if isinstance(hyper_params, dict):
        return make_hyper_string_from_dict(hyper_params)
    elif isinstance(hyper_params, list):
        return list(itertools.chain.from_iterable([make_hyper_string_from_dict(d) for d in hyper_params]))
    else:
        raise ValueError("hyper_params must be either a single dictionary or a list of dictionaries")


def submit_job(hyper, job_name):

    job_dir = Path(NAMED_EXPERIMENT_DIR) / Path(RESULTS_DIR) / job_name
    hyper['output_dir'] = str(job_dir)

    job_dir.mkdir(parents=True, exist_ok=False)

    args_str = ' '.join([f"--{k}={v}" for k, v in hyper.items()])
    python_cmd = f"{PYTHON_PATH} {SCRIPT_NAME} {args_str}"
    sys_cmd = f'echo "{python_cmd}" >> {PROJECT_DIR}/gpu.queue'
    os.system(sys_cmd)