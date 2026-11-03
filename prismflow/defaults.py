COMMON = {
    'seed': 12345,
    'model': {
        'target': 'prismflow.model.PrismFlow',
        'params': {
            'seq_length': 24,
            'n_heads': 4,
            'delta_init': 1.0e-4,
            'delta_learnable': True,
            'delta_min': 0.0,
            'lambda_power': 2.0,
            'warmup_ratio': 0.2,
        },
    },
    'dataloader': {
        'shuffle': True,
    },
    'solver': {
        'base_lr': 1.0e-4,
        'ema': {'decay': 0.995, 'update_interval': 10},
        'gradient_accumulate_every': 1,
        'scheduler': {
            'target': 'prismflow.lr_scheduler.ReduceLROnPlateauWithWarmup',
            'params': {'factor': 0.5, 'min_lr': 1.0e-5, 'patience': 2000, 'warmup': 500, 'warmup_lr': 8.0e-4},
        },
    },
}

DATASETS = {
    'sines': {
        'model': {'params': {'feature_size': 5, 'mlp_hidden_times': 4}},
        'dataloader': {
            'train_dataset': {'target': 'prismflow.data.datasets.SineDataset', 'params': {'dim': 5, 'num': 10000}},
        },
        'solver': {
            'gradient_accumulate_every': 2,
            'scheduler': {'params': {'patience': 1000, 'threshold': 0.01, 'threshold_mode': 'rel', 'warmup': 200,
                                     'warmup_lr': 5.0e-4}},
        },
    },
    'stocks': {
        'seed': 2,
        'model': {'params': {'feature_size': 6, 'mlp_hidden_times': 2, 'lambda_power': 4.0}},
        'dataloader': {
            'train_dataset': {'target': 'prismflow.data.datasets.CustomDataset',
                              'params': {'data_root': './datasets/stock_data.csv', 'name': 'stock'}},
        },
    },
    'etth': {
        'model': {'params': {'feature_size': 7, 'mlp_hidden_times': 4}},
        'dataloader': {
            'train_dataset': {'target': 'prismflow.data.datasets.CustomDataset',
                              'params': {'data_root': './datasets/ETTh.csv', 'name': 'etth'}},
        },
        'solver': {
            'gradient_accumulate_every': 2,
            'scheduler': {'params': {'patience': 4000, 'threshold': 0.01, 'threshold_mode': 'rel',
                                     'warmup_lr': 5.0e-4}},
        },
    },
    'mujoco': {
        'model': {'params': {'feature_size': 14, 'mlp_hidden_times': 1}},
        'dataloader': {
            'train_dataset': {'target': 'prismflow.data.datasets.MuJoCoDataset', 'params': {'dim': 14, 'num': 10000}},
        },
    },
    'energy': {
        'model': {'params': {'feature_size': 28, 'mlp_hidden_times': 4}},
        'dataloader': {
            'train_dataset': {'target': 'prismflow.data.datasets.CustomDataset',
                              'params': {'data_root': './datasets/energy_data.csv', 'name': 'energy'}},
        },
        'solver': {'gradient_accumulate_every': 2},
    },
    'fmri': {
        'model': {'params': {'feature_size': 50, 'mlp_hidden_times': 4, 'prediction': 'hybrid', 'noise_scale': 1.1}},
        'dataloader': {
            'train_dataset': {'target': 'prismflow.data.datasets.fMRIDataset',
                              'params': {'data_root': './datasets/fMRI', 'name': 'fMRI'}},
        },
        'solver': {'gradient_accumulate_every': 2},
    },
}
