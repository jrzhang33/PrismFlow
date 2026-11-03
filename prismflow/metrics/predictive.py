import numpy as np
import torch


class _Predictor(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.rnn = torch.nn.GRU(input_size=dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return torch.sigmoid(self.linear(out))


def predictive_score(real, fake, iterations=5000, batch_size=128, device='cuda'):
    dim = real.shape[-1]
    model = _Predictor(dim - 1, int(dim / 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for _ in range(iterations):
        mb = fake[np.random.choice(len(fake), batch_size, replace=True)]
        x = torch.from_numpy(mb[:, :-1, :dim - 1]).float().to(device)
        y = torch.from_numpy(mb[:, 1:, dim - 1:]).float().to(device)
        loss = torch.nn.functional.l1_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    abs_err = []
    with torch.no_grad():
        for start in range(0, len(real), 4096):
            chunk = real[start:start + 4096]
            pred = model(torch.from_numpy(chunk[:, :-1, :dim - 1]).float().to(device)).cpu().numpy()
            abs_err.append(np.abs(pred - chunk[:, 1:, dim - 1:]).mean(axis=(1, 2)))
    return float(np.concatenate(abs_err).mean())
