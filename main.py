import argparse
import os

import numpy as np
import torch

from prismflow.data import build_dataloader
from prismflow.trainer import Trainer
from prismflow.utils import Logger, instantiate_from_config, load_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='path to a configs/<dataset>.yaml file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='train from scratch')
    group.add_argument('--sample', action='store_true', help='sample from a checkpoint')
    parser.add_argument('--ckpt_dir', default=None, help='checkpoint folder (default: solver.results_folder)')
    parser.add_argument('--milestone', type=int, default=10, help='which checkpoint-<milestone>.pt to load')
    parser.add_argument('--output', default=None, help='output folder (default: OUTPUT/<config name>)')
    parser.add_argument('--seed', type=int, default=None, help='overrides the seed in the config')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('overrides', nargs='*', help='config overrides as key.path=value')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed = args.seed if args.seed is not None else config.get('seed', 12345)
    save_dir = args.output or os.path.join('OUTPUT', config['dataset'])
    ckpt_dir = args.ckpt_dir or config['solver']['results_folder']

    seed_everything(seed)
    torch.cuda.set_device(args.gpu)
    logger = Logger(save_dir, config)
    logger.log_info(f'seed {seed}, checkpoints in {ckpt_dir}')

    model = instantiate_from_config(config['model']).cuda()
    train_loader, dataset = build_dataloader(config)
    trainer = Trainer(config, model, train_loader, ckpt_dir, logger=logger)

    if args.train:
        trainer.train()
        return

    trainer.load(args.milestone)
    samples = trainer.sample(num=len(dataset.reference))
    np.save(os.path.join(save_dir, 'fake.npy'), dataset.to_eval_space(samples))
    np.save(os.path.join(save_dir, 'real.npy'), dataset.reference)
    logger.log_info(f'saved {samples.shape} samples to {save_dir}/fake.npy')


if __name__ == '__main__':
    main()
