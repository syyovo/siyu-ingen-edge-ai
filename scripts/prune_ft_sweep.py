"""
Week 2 - Structured pruning sweep with proper recovery budget.

Why this script: the notebook sweep (1 epoch x 5k images) under-recovers -
10% pruning reached only 54%, 20%+ stayed at random. That measures the
recovery budget, not the model's true prunability. This run gives every
ratio 3 epochs x 20k images so the accuracy-vs-ratio curve reflects the
architecture, and PELT has a real cliff to detect.

Usage:
    python scripts/prune_ft_sweep.py               # full overnight run (~5h)
    python scripts/prune_ft_sweep.py --smoke-test  # 1 ratio, tiny budget (~10 min)

Output:
    experiments/W02_pruning_sweep_ft3.csv   (updated after every ratio)
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_pruning as tp
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAIN_TFM = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
TEST_TFM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class HFCifar10(Dataset):
    def __init__(self, hf_split, tfm):
        self.ds, self.tfm = hf_split, tfm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.tfm(item['img']), item['label']


def build_model():
    m = mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(1024, 10)
    return m


def load_baseline():
    m = build_model()
    m.load_state_dict(torch.load('checkpoints/baseline_best.pth',
                                 map_location='cpu'))
    m.eval()
    return m


def prune_model(ratio):
    model = load_baseline()
    example = torch.randn(1, 3, 224, 224)
    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=importance,
        pruning_ratio=ratio,
        ignored_layers=[model.classifier[3]],
    )
    pruner.step()
    return model


def finetune(model, loader, epochs, lr=1e-4):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        running, n = 0.0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        print(f'    ft epoch {ep}/{epochs} mean loss {running / n:.4f}',
              flush=True)
    model.eval()
    return model


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def model_size_mb(model):
    tmp = Path('experiments/_tmp_size.pth')
    tmp.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), tmp)
    mb = tmp.stat().st_size / 1e6
    tmp.unlink()
    return mb


def measure_latency_ms(model, warmup=10, runs=100):
    x = torch.randn(1, 3, 224, 224)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    ds = load_dataset('uoft-cs/cifar10')
    ft_hf = ds['train'].shuffle(seed=SEED).select(
        range(2000 if args.smoke_test else 20000))
    test_hf = ds['test'].select(range(1000)) if args.smoke_test else ds['test']
    epochs = 1 if args.smoke_test else 3
    ratios = [0.1] if args.smoke_test else \
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ft_loader = DataLoader(HFCifar10(ft_hf, TRAIN_TFM), batch_size=64,
                           shuffle=True, num_workers=4,
                           persistent_workers=True)
    test_loader = DataLoader(HFCifar10(test_hf, TEST_TFM), batch_size=128,
                             shuffle=False, num_workers=4,
                             persistent_workers=True)

    out_csv = Path('experiments/W02_pruning_sweep_ft3.csv')
    base = load_baseline()
    rows = [{'config': 'baseline', 'ratio': 0.0,
             'params': sum(p.numel() for p in base.parameters()),
             'test_acc': round(evaluate(base, test_loader), 4),
             'size_MB': round(model_size_mb(base), 2)}]
    lat, std = measure_latency_ms(base)
    rows[0].update({'latency_ms': round(lat, 2), 'latency_std_ms': round(std, 2)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(rows[0], flush=True)

    for r in ratios:
        t0 = time.time()
        print(f'[sweep] ratio {r}: pruning + {epochs}ep finetune...', flush=True)
        m = finetune(prune_model(r), ft_loader, epochs)
        row = {'config': f'prune {int(r*100)}% + {epochs}ep ft', 'ratio': r,
               'params': sum(p.numel() for p in m.parameters()),
               'test_acc': round(evaluate(m, test_loader), 4),
               'size_MB': round(model_size_mb(m), 2)}
        lat, std = measure_latency_ms(m)
        row.update({'latency_ms': round(lat, 2),
                    'latency_std_ms': round(std, 2)})
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'[sweep] {row} | {(time.time()-t0)/60:.1f} min', flush=True)

    print('DONE. Results in', out_csv, flush=True)


if __name__ == '__main__':
    main()
