import argparse
import json
import os
import random

import numpy as np
import scipy.stats
import torch

from prismflow.metrics.context_fid import context_fid
from prismflow.metrics.correlation import correlational_score
from prismflow.metrics.discriminative import discriminative_score
from prismflow.metrics.predictive import predictive_score


def summarize(scores):
    mean = float(np.mean(scores))
    ci95 = float(scipy.stats.sem(scores) * scipy.stats.t.ppf((1 + 0.95) / 2.0, len(scores) - 1))
    return mean, ci95


def correlational_iterations(real, fake, iterations):
    x_real, x_fake = torch.from_numpy(real), torch.from_numpy(fake)
    size = max(1, x_real.shape[0] // iterations)
    scores = []
    for _ in range(iterations):
        real_idx = torch.randint(0, x_real.shape[0], (size,))
        fake_idx = torch.randint(0, x_fake.shape[0], (size,))
        scores.append(correlational_score(x_real[real_idx], x_fake[fake_idx]))
    return scores


def plot_all(real, fake, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    real_m, fake_m = real.mean(-1), fake.mean(-1)
    pca = PCA(n_components=2).fit(real_m)
    try:
        tsne = TSNE(n_components=2, perplexity=40, max_iter=300)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    emb = tsne.fit_transform(np.vstack([real_m, fake_m]))
    views = {
        'pca': (pca.transform(real_m), pca.transform(fake_m)),
        'tsne': (emb[:len(real_m)], emb[len(real_m):]),
    }
    for name, (r, f) in views.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(r[:, 0], r[:, 1], c='red', alpha=0.3, label='Real', s=10)
        ax.scatter(f[:, 0], f[:, 1], c='blue', alpha=0.3, label='Generated', s=10)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(real_m.flatten(), linewidth=3, label='Real', color='red', ax=ax)
    sns.kdeplot(fake_m.flatten(), linewidth=3, linestyle='--', label='Generated', color='blue', ax=ax)
    ax.set_xlabel('Data Value')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'kernel.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--real', required=True)
    parser.add_argument('--fake', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0, help='seeds TS2Vec, the subsets and the post-hoc GRUs')
    parser.add_argument('--metrics', default='context_fid,correlational,discriminative,predictive')
    parser.add_argument('--plots', action='store_true', help='also save PCA / t-SNE / KDE plots')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    real, fake = np.load(args.real), np.load(args.fake)
    n = min(len(real), len(fake))
    real, fake = real[:n], fake[:n]
    print(f'real {real.shape}, fake {fake.shape}')

    runners = {
        'context_fid': lambda: [context_fid(real, fake) for _ in range(args.iterations)],
        'correlational': lambda: correlational_iterations(real, fake, args.iterations),
        'discriminative': lambda: [discriminative_score(real, fake) for _ in range(args.iterations)],
        'predictive': lambda: [predictive_score(real, fake) for _ in range(args.iterations)],
    }
    metrics = {'n_sequences': int(n), 'iterations': args.iterations, 'seed': args.seed}
    for name in args.metrics.split(','):
        scores = [float(s) for s in runners[name]()]
        mean, ci95 = summarize(scores)
        metrics.update({f'{name}_scores': scores, f'{name}_mean': mean, f'{name}_ci95': ci95})
        print(f'{name:15s} {mean:.4f} ± {ci95:.4f}   runs: ' + ', '.join(f'{s:.4f}' for s in scores))

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    if args.plots:
        plot_all(real, fake, args.out_dir)
    print(f'saved {args.out_dir}/metrics.json')


if __name__ == '__main__':
    main()
