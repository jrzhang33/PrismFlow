# PrismFlow: Residual Dynamics for Flow Matching in Time-Series Generation

[![Paper](https://img.shields.io/badge/arXiv-2605.28867-b31b1b.svg)](https://arxiv.org/pdf/2605.28867)

Paper: [https://arxiv.org/pdf/2605.28867](https://arxiv.org/pdf/2605.28867)

Official implementation of PrismFlow (NeurIPS 2026). PrismFlow augments a flow-matching generator with Koopman-inspired dynamical experts. A time-aware router selects one expert for each flow state. A confidence-aware winner-take-all objective trains the experts. During sampling, the selected expert adds a residual correction to the global transport field.

## Installation

```bash
conda create -n prismflow python=3.10 -y
conda activate prismflow
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
export MUJOCO_GL=egl
```

## Data

We use the datasets provided by [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS?tab=readme-ov-file). Download them from the link in that repository and place the files under `datasets/`:

```
datasets/
├── stock_data.csv
├── ETTh.csv
├── energy_data.csv
└── fMRI/
    └── sim4.mat
```

Sines and MuJoCo are generated on the fly and need no download.

## Usage

Training writes the checkpoints to `checkpoints/<dataset>/`. Sampling loads the final checkpoint from the same folder and writes the generated and real sequences to `OUTPUT/<dataset>/`, and evaluation writes the scores to `results/<dataset>/metrics.json`.

### Sines

```bash
python main.py --config configs/sines.yaml --train
python main.py --config configs/sines.yaml --sample
python evaluate.py --real OUTPUT/sines/real.npy --fake OUTPUT/sines/fake.npy --out_dir results/sines
```

### Stocks

```bash
python main.py --config configs/stocks.yaml --train
python main.py --config configs/stocks.yaml --sample
python evaluate.py --real OUTPUT/stocks/real.npy --fake OUTPUT/stocks/fake.npy --out_dir results/stocks
```

### ETTh

```bash
python main.py --config configs/etth.yaml --train
python main.py --config configs/etth.yaml --sample
python evaluate.py --real OUTPUT/etth/real.npy --fake OUTPUT/etth/fake.npy --out_dir results/etth
```

### MuJoCo

```bash
python main.py --config configs/mujoco.yaml --train
python main.py --config configs/mujoco.yaml --sample
python evaluate.py --real OUTPUT/mujoco/real.npy --fake OUTPUT/mujoco/fake.npy --out_dir results/mujoco
```

### Energy

```bash
python main.py --config configs/energy.yaml --train
python main.py --config configs/energy.yaml --sample
python evaluate.py --real OUTPUT/energy/real.npy --fake OUTPUT/energy/fake.npy --out_dir results/energy
```

### fMRI

```bash
python main.py --config configs/fmri.yaml --train
python main.py --config configs/fmri.yaml --sample
python evaluate.py --real OUTPUT/fmri/real.npy --fake OUTPUT/fmri/fake.npy --out_dir results/fmri
```

## Contact

For questions, please contact Junru Zhang (junruzhang@zju.edu.cn).

## Acknowledgement

We appreciate the following GitHub repository for its valuable codebase and datasets:

- [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS)

## Citation

```bibtex
@inproceedings{zhang2026prismflow,
  title     = {PrismFlow: Residual Dynamics for Flow Matching in Time-Series Generation},
  author    = {Zhang, Junru and Feng, Lang and Wang, Jinbo and Guo, Xu and Wang, Yucheng and Yu, Han and Wu, Min and Dong, Yabo and Xu, Duanqing},
  booktitle = {The 40th Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```
