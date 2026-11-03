import os

import numpy as np
import pandas as pd
import torch
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


class TimeSeriesDataset(Dataset):
    def to_eval_space(self, x):
        return unnormalize_to_zero_to_one(x) if self.auto_norm else x

    def __getitem__(self, ind):
        return torch.from_numpy(self.samples[ind]).float()

    def __len__(self):
        return self.samples.shape[0]


class SineDataset(TimeSeriesDataset):
    def __init__(self, window=24, num=10000, dim=5, neg_one_to_one=True, seed=123):
        super().__init__()
        self.window, self.var_num, self.auto_norm = window, dim, neg_one_to_one
        self.rawdata = self.sine_data_generation(num, window, dim, seed)
        self.samples = normalize_to_neg_one_to_one(self.rawdata) if neg_one_to_one else self.rawdata
        self.reference = self.rawdata

    @staticmethod
    def sine_data_generation(no, seq_len, dim, seed=123):
        st0 = np.random.get_state()
        np.random.seed(seed)
        data = []
        for _ in tqdm(range(no), desc='Sampling sine-dataset', leave=False):
            temp = []
            for _ in range(dim):
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)
                temp.append([np.sin(freq * j + phase) for j in range(seq_len)])
            data.append((np.transpose(np.asarray(temp)) + 1) * 0.5)
        np.random.set_state(st0)
        return np.array(data)


class CustomDataset(TimeSeriesDataset):
    def __init__(self, name, data_root, window=24, neg_one_to_one=True, seed=123):
        super().__init__()
        self.name, self.window, self.auto_norm = name, window, neg_one_to_one
        rawdata, self.scaler = self.read_data(data_root, name)
        self.var_num = rawdata.shape[-1]

        data = self.scaler.transform(rawdata)
        if neg_one_to_one:
            data = normalize_to_neg_one_to_one(data)
        n_windows = max(rawdata.shape[0] - window + 1, 0)
        windows = np.zeros((n_windows, window, self.var_num))
        for i in range(n_windows):
            windows[i] = data[i:i + window]
        self.samples = windows
        self.reference = self.to_eval_space(windows)

    @staticmethod
    def read_data(filepath, name=''):
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        return data, MinMaxScaler().fit(data)


class fMRIDataset(CustomDataset):
    @staticmethod
    def read_data(filepath, name=''):
        data = io.loadmat(os.path.join(filepath, 'sim4.mat'))['ts']
        return data, MinMaxScaler().fit(data)


class MuJoCoDataset(TimeSeriesDataset):
    def __init__(self, window=24, num=10000, dim=14, neg_one_to_one=True, seed=123):
        super().__init__()
        self.window, self.var_num, self.auto_norm = window, dim, neg_one_to_one
        rawdata, self.scaler = self._generate_random_trajectories(num, seed)
        data = self.scaler.transform(rawdata.reshape(-1, dim))
        if neg_one_to_one:
            data = normalize_to_neg_one_to_one(data)
        self.samples = data.reshape(-1, window, dim)
        self.reference = self.to_eval_space(self.samples)

    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite
        except ImportError as e:
            raise ImportError('dm_control is required to generate the MuJoCo dataset.') from e
        physics = suite.load('hopper', 'stand').physics

        st0 = np.random.get_state()
        np.random.seed(seed)
        data = np.zeros((n_samples, self.window, self.var_num))
        for i in range(n_samples):
            with physics.reset_context():
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)
            for t in range(self.window):
                data[i, t, :self.var_num // 2] = physics.data.qpos
                data[i, t, self.var_num // 2:] = physics.data.qvel
                physics.step()
        np.random.set_state(st0)
        return data, MinMaxScaler().fit(data.reshape(-1, self.var_num))
