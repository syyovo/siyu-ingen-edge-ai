"""
Week 2 - Quantization-Aware Training (QAT)
Recovers the accuracy that PTQ lost on MobileNetV3-Small.

Why QAT: all PTQ configs collapsed (15.9% / 12.2% / 10.0% vs 93.38% baseline)
because quantization error accumulates across 50+ layers of a compact
depthwise-separable network. QAT inserts fake-quantize nodes during training:
the forward pass rounds values to INT8 granularity, gradients flow through,
so weights learn to sit where quantization hurts least (Nagel et al., sec. QAT).

Usage:
    python scripts/qat_train.py               # full run, 3 epochs (~3-4h CPU)
    python scripts/qat_train.py --smoke-test  # 2k images, 1 epoch (~10 min)

Outputs:
    checkpoints/qat_int8_state.pth   - converted INT8 model state_dict
    checkpoints/qat_summary.json     - final metrics
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import quantize_fx
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.quantized.engine = 'qnnpack'

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


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    ckpt_dir = Path('checkpoints')

    print('Loading CIFAR-10...', flush=True)
    ds = load_dataset('uoft-cs/cifar10')
    train_hf, test_hf = ds['train'], ds['test']
    epochs = args.epochs
    if args.smoke_test:
        train_hf = train_hf.select(range(2000))
        test_hf = test_hf.select(range(1000))
        epochs = 1

    # num_workers=4 is fine here: classes live in a real .py file,
    # so macOS spawn subprocesses can import them (unlike in notebooks).
    train_loader = DataLoader(HFCifar10(train_hf, TRAIN_TFM), batch_size=64,
                              shuffle=True, num_workers=4,
                              persistent_workers=True)
    test_loader = DataLoader(HFCifar10(test_hf, TEST_TFM), batch_size=128,
                             shuffle=False, num_workers=4,
                             persistent_workers=True)

    # Start from the fine-tuned FP32 baseline (93.38%) - QAT is a short
    # adaptation on top of good weights, not training from scratch.
    model = build_model()
    model.load_state_dict(torch.load('checkpoints/baseline_best.pth',
                                     map_location='cpu'))

    # QAT config: per-channel weights (same protective choice as PTQ),
    # MovingAverage observers inside FakeQuantize so ranges adapt smoothly
    # while weights shift during training.
    qat_qconfig = tq.QConfig(
        activation=tq.FakeQuantize.with_args(
            observer=tq.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8,
            reduce_range=False),
        weight=tq.FakeQuantize.with_args(
            observer=tq.MovingAveragePerChannelMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric),
    )
    qmap = tq.QConfigMapping().set_global(qat_qconfig)
    example_inputs = (torch.randn(1, 3, 224, 224),)

    # prepare_qat_fx inserts FakeQuantize nodes; model must be in train mode.
    model.train()
    prepared = quantize_fx.prepare_qat_fx(model, qmap, example_inputs)

    criterion = nn.CrossEntropyLoss()
    # Small lr: weights only need to shift slightly into quantization-
    # friendly positions; a large lr would destroy the 93.38% starting point.
    optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-4)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        prepared.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = criterion(prepared(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 50 == 0:
                print(f'[qat] epoch {ep}/{epochs} batch {i}/{len(train_loader)} '
                      f'loss {running / 50:.4f}', flush=True)
                running = 0.0

        # Convert a COPY to true INT8 each epoch and track the best one -
        # what we care about is INT8 accuracy, not fake-quant accuracy.
        prepared.apply(tq.disable_observer)
        fq_acc = evaluate(prepared, test_loader)
        prepared.apply(tq.enable_observer)
        print(f'[qat] epoch {ep} fake-quant (simulated INT8) acc {fq_acc:.4f}',
              flush=True)
        
        int8 = quantize_fx.convert_fx(copy.deepcopy(prepared).eval())
        acc = evaluate(int8, test_loader)
        mins = (time.time() - t0) / 60
        print(f'[qat] epoch {ep}/{epochs} done in {mins:.1f} min | '
              f'INT8 test acc {acc:.4f}', flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(int8.state_dict(), ckpt_dir / 'qat_int8_state.pth')
            print(f'[qat] new best INT8 acc {best_acc:.4f} -> saved', flush=True)

    with open(ckpt_dir / 'qat_summary.json', 'w') as f:
        json.dump({'seed': SEED, 'epochs': epochs,
                   'best_int8_test_acc': round(best_acc, 4),
                   'smoke_test': args.smoke_test}, f, indent=2)
    print(f'DONE. Best INT8 test accuracy after QAT: {best_acc:.4f} '
          f'(vs 0.9338 FP32 baseline, vs 0.1591 PTQ)', flush=True)


if __name__ == '__main__':
    main()