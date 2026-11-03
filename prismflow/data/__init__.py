import torch

from ..utils import instantiate_from_config


def build_dataloader(config):
    cfg = config['dataloader']
    dataset = instantiate_from_config(cfg['train_dataset'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=0,
                                         pin_memory=True, drop_last=cfg['shuffle'])
    return loader, dataset
