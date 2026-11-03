import torch


def cacf_torch(x, max_lag, dim=(0, 1)):
    ind = [list(i) for i in torch.tril_indices(x.shape[2], x.shape[2])]
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)
    x_l, x_r = x[..., ind[0]], x[..., ind[1]]
    cacf = [torch.mean(x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r, 1) for i in range(max_lag)]
    cacf = torch.cat(cacf, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def correlational_score(x_real, x_fake):
    real = cacf_torch(x_real, 1).mean(0)[0]
    fake = cacf_torch(x_fake, 1).mean(0)[0]
    return (torch.abs(fake - real).sum(0) / 10.).item()
