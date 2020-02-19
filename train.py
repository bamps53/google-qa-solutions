import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import wandb
import os
import warnings
import gc
import torch
from apex import amp
warnings.filterwarnings("ignore")

from config.base import load_config
from models import get_model
from utils import seed_everything
from utils.loops import train, Runner
from utils.learner import Learner
from utils.metrics import compute_spearmanr_ignore_nan
from datasets import get_train_val_loaders
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler

def run(args, idx_fold):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    print('use gpu No.{}'.format(args.device_id))

    config = load_config(args.config_path)
    root = Path(config.work_dir)
    os.makedirs(root,exist_ok=True)
    exp_name = config.work_dir.split('/')[-1]

    seed_everything(config.seed)

    wandb.init(project=config.project_name, name='{}_fold{}'.format(exp_name, idx_fold), config=config, reinit=True)

    loaders = get_train_val_loaders(config, idx_fold=idx_fold, debug=args.debug)

    # 1. train only classifier
    model = get_model(config)
    wandb.watch(model, log="all")

    optimizer = get_optimizer(config, model, train_only_head=True)
    scheduler = get_scheduler(config, optimizer, config.train.warmup_num_epochs, num_batches=len(loaders['train']))

    learner = Learner(
        config=config,
        model=model,
        criterion=get_loss(config),
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        metric_fn=compute_spearmanr_ignore_nan,
        fold=idx_fold,
        train_only_head=True
        )
    print('start training only classfier.')
    learner.train()

    # 2. fine tune all
    optimizer = get_optimizer(config, model, train_only_head=False)
    scheduler = get_scheduler(config, optimizer, config.train.num_epochs, num_batches=len(loaders['train']))

    learner = Learner(
        config=config,
        model=model,
        criterion=get_loss(config),
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        metric_fn=compute_spearmanr_ignore_nan,
        fold=idx_fold,
        train_only_head=False
        )
    print('start training full model.')
    learner.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str)
    parser.add_argument('--device_id', '-d', default='0', type=str)
    parser.add_argument('--start_fold', '-s', default=0, type=int)
    parser.add_argument('--end_fold', '-e', default=4, type=int)
    parser.add_argument('--debug', '-db', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for idx_fold in range(args.start_fold, args.end_fold+1):
        run(args, idx_fold)
