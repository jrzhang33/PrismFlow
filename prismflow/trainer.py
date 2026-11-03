import os
import time
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm.auto import tqdm

from .utils import count_parameters, instantiate_from_config


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer:
    def __init__(self, config, model, dataloader, ckpt_dir, logger=None):
        self.model = model
        self.device = model.betas.device
        solver = config['solver']
        self.train_num_steps = solver['max_epochs']
        self.gradient_accumulate_every = solver['gradient_accumulate_every']
        self.save_cycle = solver['save_cycle']
        self.dl = cycle(dataloader) if dataloader is not None else None
        self.step = 0
        self.milestone = 0
        self.logger = logger
        self.log_frequency = 1000

        self.results_folder = Path(ckpt_dir)
        os.makedirs(self.results_folder, exist_ok=True)

        self.opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=solver.get('base_lr', 1.0e-4),
                        betas=[0.9, 0.96])
        self.ema = EMA(model, beta=solver['ema']['decay'], update_every=solver['ema']['update_interval']).to(
            self.device)
        sc_cfg = {'target': solver['scheduler']['target'], 'params': dict(solver['scheduler']['params'])}
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if logger is not None:
            logger.log_info(f'trainable parameters: {count_parameters(model):,}')

    def save(self, milestone):
        torch.save({
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone):
        path = self.results_folder / f'checkpoint-{milestone}.pt'
        data = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        ema_state = data['ema']
        for key in ('initted', 'step'):
            if key in ema_state and ema_state[key].dim() == 0:
                ema_state[key] = ema_state[key].unsqueeze(0)
        self.ema.load_state_dict(ema_state)
        self.milestone = milestone
        if self.logger is not None:
            self.logger.log_info(f'loaded {path} (step {self.step})')

    def train(self):
        tic = time.time()
        with tqdm(total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(self.device)
                    loss = self.model(data) / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()
                pbar.set_description(f'loss: {total_loss:.6f}')

                grad_norm = clip_grad_norm_(self.model.parameters(), 1.0)
                if torch.isfinite(grad_norm):
                    self.opt.step()
                elif self.logger is not None:
                    self.logger.log_info(f'step {self.step}: non-finite gradient, update skipped')
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                self.model._training_step.fill_(self.step)
                self.ema.update()

                if self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)
                if self.logger is not None and self.step % self.log_frequency == 0:
                    self.logger.log_info(f'step {self.step}: loss {total_loss:.6f}')
                pbar.update(1)
        if self.logger is not None:
            self.logger.log_info(f'training done in {time.time() - tic:.1f}s')

    @torch.no_grad()
    def sample(self, num, size_every=2001):
        tic = time.time()
        model = self.ema.ema_model
        samples = []
        for _ in range(num // size_every + 1):
            samples.append(model.generate(size_every).cpu().numpy())
            torch.cuda.empty_cache()
        if self.logger is not None:
            self.logger.log_info(f'sampling done in {time.time() - tic:.2f}s')
        return np.concatenate(samples, axis=0)[:num].astype(np.float64)

