import copy
import importlib
import os
import random
import sys
import time

import numpy as np
import torch
import yaml


from .defaults import COMMON, DATASETS


def deep_merge(base, update):
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path, overrides=()):
    with open(path) as f:
        paper_config = yaml.safe_load(f)
    name = paper_config.get('dataset') or os.path.splitext(os.path.basename(path))[0]
    config = deep_merge(deep_merge(COMMON, DATASETS.get(name, {})), paper_config)
    config['dataset'] = name
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep:
            raise ValueError(f'override must look like key.path=value, got {item!r}')
        node = config
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)

    solver = config['solver']
    solver.setdefault('results_folder', f'./checkpoints/{name}')
    solver.setdefault('save_cycle', max(1, solver['max_epochs'] // 10))
    config['model']['params'].setdefault('max_epochs', solver['max_epochs'])
    return config


def instantiate_from_config(config):
    module, cls = config['target'].rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)(**config.get('params', dict()))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger:
    def __init__(self, save_dir, config=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'command.txt'), 'a') as f:
            f.write(f'torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}\n')
            f.write(' '.join(sys.argv) + '\n')
        if config is not None:
            with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
                yaml.safe_dump(config, f, sort_keys=False)
        self.text_writer = open(os.path.join(save_dir, 'log.txt'), 'a')

    def log_info(self, info):
        print(info)
        self.text_writer.write(f"{time.strftime('%Y-%m-%d-%H-%M')}: {info}\n")
        self.text_writer.flush()
