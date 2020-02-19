import random
import numpy as np
import torch
from logzero import logger
import os
import json

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info('Set random seed={}'.format(seed))

def dict_to_json(dict_obj, file_name):
    with open(file_name, 'w') as fp:
        json.dump(dict_obj, fp)

def to_cpu(x):
    return x.contiguous().detach().cpu()


def to_numpy(x):
    return to_cpu(x).numpy()


def to_device(xs, device, no_wrap_for_singles=False):
    if isinstance(xs, tuple) or isinstance(xs, list):
        return [to_device(x, device, no_wrap_for_singles=True) for x in xs]
    elif isinstance(xs, dict):
        return {k:to_device(v, device, no_wrap_for_singles=True) for k, v in xs.items()}
        # for k, v in xs.items():
        #     v = to_device(v, device, no_wrap_for_singles=True)
        # for k, v in xs.items():
        #     print(v.device)
        #     print(device)
        # return xs
    else:
        if no_wrap_for_singles: return xs.to(device)
        else: return [xs.to(device)]


def set_optimizer_mom(opt, mom):
    has_betas = 'betas' in opt.param_groups[0]
    has_mom = 'momentum' in opt.param_groups[0]
    if has_betas:
        for g in opt.param_groups:
            _, beta = g['betas']
            g['betas'] = (mom, beta)
    elif has_mom:
        for g in opt.param_groups:
            g['momentum'] = mom