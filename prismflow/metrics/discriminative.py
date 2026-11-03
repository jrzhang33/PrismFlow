import numpy as np
import torch
from sklearn.metrics import accuracy_score


class _Discriminator(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.rnn = torch.nn.GRU(input_size=dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.linear(h.squeeze(0))


def _split(data, train_rate=0.8):
    idx = np.random.permutation(len(data))
    n = int(len(data) * train_rate)
    return data[idx[:n]], data[idx[n:]]


def discriminative_score(real, fake, iterations=2000, batch_size=128, device='cuda'):
    dim = real.shape[-1]
    model = _Discriminator(dim, int(dim / 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_x, test_x = _split(real)
    train_x_hat, test_x_hat = _split(fake)

    model.train()
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(iterations):
        x_mb = torch.from_numpy(train_x[np.random.choice(len(train_x), batch_size, replace=True)]).float().to(device)
        x_hat_mb = torch.from_numpy(
            train_x_hat[np.random.choice(len(train_x_hat), batch_size, replace=True)]).float().to(device)
        logit_real, logit_fake = model(x_mb), model(x_hat_mb)
        loss = bce(logit_real, torch.ones_like(logit_real)) + bce(logit_fake, torch.zeros_like(logit_fake))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        p_real = torch.sigmoid(model(torch.from_numpy(test_x).float().to(device))).cpu().numpy()
        p_fake = torch.sigmoid(model(torch.from_numpy(test_x_hat).float().to(device))).cpu().numpy()
    y_pred = np.squeeze(np.concatenate((p_real, p_fake), axis=0))
    y_true = np.concatenate((np.ones(len(p_real)), np.zeros(len(p_fake))))
    return float(np.abs(0.5 - accuracy_score(y_true, y_pred > 0.5)))
