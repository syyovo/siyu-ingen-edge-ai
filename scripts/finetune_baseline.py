"""
Week 2 - Baseline fine-tuning script
Fine-tunes ImageNet-pretrained MobileNetV3-Small on CIFAR-10 (CPU only).

Why this exists: Week 1 profiled the ImageNet-head model (no CIFAR-10 accuracy).
Every Week 2 compression experiment needs a "baseline accuracy" for the
8-element results table (Mars Trading standard), so we must first produce
a CIFAR-10-trained baseline model.

Usage:
    python scripts/finetune_baseline.py               # full run (~2-3.5h on Apple M CPU)
    python scripts/finetune_baseline.py --smoke-test  # quick pipeline check (~10 min)

Outputs:
    checkpoints/baseline_best.pth   - best test-accuracy weights (Week 2 baseline)
    checkpoints/baseline_last.pth   - last-epoch weights (crash recovery)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ---------------------------------------------------------------- reproducibility
# Project plan section 5.2: every experiment must reproduce from a clean
# run with the same seeds. Seed all three RNG sources PyTorch code touches.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # project constraint: CPU-only experiments

# ImageNet normalization stats - the pretrained weights were trained with
# these, so inputs must be normalized the same way or features degrade.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TFM = transforms.Compose([
    transforms.Resize(224),              # match pretrained input size & W1 latency setup
    transforms.RandomHorizontalFlip(),   # light augmentation to reduce overfitting
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
TEST_TFM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class HFCifar10(Dataset):
    """Thin wrapper: HuggingFace dataset -> PyTorch Dataset with transforms.

    Keeps the whole repo on one data source (uoft-cs/cifar10, same as W1)
    so accuracy numbers are comparable across all notebooks.
    """

    def __init__(self, hf_split, tfm):
        self.ds = hf_split
        self.tfm = tfm

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.tfm(item["img"]), item["label"]


def build_model():
    """Load pretrained MobileNetV3-Small and swap the 1000-class ImageNet
    head for a 10-class CIFAR-10 head.

    Note: this shrinks the model (~2.54M -> ~1.53M params, ~10.3MB -> ~5.8MB)
    because the old head alone held ~1M params. Re-profile all four baseline
    metrics on THIS model before quantization/pruning experiments.
    """
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features  # 1024
    model.classifier[3] = nn.Linear(in_features, 10)
    return model


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def train_epochs(model, train_loader, test_loader, epochs, lr, tag, best_acc, ckpt_dir):
    """One training phase. Returns updated best accuracy."""
    criterion = nn.CrossEntropyLoss()
    # Only pass trainable params to the optimizer (frozen ones have no grad).
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 50 == 0:
                print(f"[{tag}] epoch {ep}/{epochs} batch {i}/{len(train_loader)} "
                      f"loss {running / 50:.4f}", flush=True)
                running = 0.0
        acc = evaluate(model, test_loader)
        mins = (time.time() - t0) / 60
        print(f"[{tag}] epoch {ep}/{epochs} done in {mins:.1f} min | "
              f"test acc {acc:.4f}", flush=True)
        torch.save(model.state_dict(), ckpt_dir / "baseline_last.pth")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_dir / "baseline_best.pth")
            print(f"[{tag}] new best acc {best_acc:.4f} -> saved baseline_best.pth",
                  flush=True)
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny subset + 1 epoch, verifies pipeline end-to-end")
    args = parser.parse_args()

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print("Loading CIFAR-10 from HuggingFace cache...", flush=True)
    ds = load_dataset("uoft-cs/cifar10")
    train_hf, test_hf = ds["train"], ds["test"]

    if args.smoke_test:
        # Small fixed slices: fast, deterministic, exercises every code path.
        train_hf = train_hf.select(range(2000))
        test_hf = test_hf.select(range(1000))
        phase1_epochs, phase2_epochs = 1, 0
    else:
        phase1_epochs, phase2_epochs = 2, 2

    train_loader = DataLoader(
        HFCifar10(train_hf, TRAIN_TFM), batch_size=64, shuffle=True,
        num_workers=4, persistent_workers=True,
    )
    test_loader = DataLoader(
        HFCifar10(test_hf, TEST_TFM), batch_size=128, shuffle=False,
        num_workers=4, persistent_workers=True,
    )

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model ready | params after head swap: {n_params:,}", flush=True)

    best_acc = 0.0

    # Phase 1: freeze backbone, train only the new head.
    # The pretrained features are already good; the randomly initialized head
    # is not. Training the head alone first avoids large early gradients
    # wrecking the pretrained features, and each step is cheaper.
    for p in model.features.parameters():
        p.requires_grad = False
    print(f"Phase 1: head-only training, {phase1_epochs} epoch(s), lr=1e-3", flush=True)
    best_acc = train_epochs(model, train_loader, test_loader,
                            phase1_epochs, 1e-3, "phase1", best_acc, ckpt_dir)

    # Phase 2: unfreeze everything, fine-tune end-to-end at 10x smaller lr
    # so the pretrained features are adjusted gently, not overwritten.
    if phase2_epochs > 0:
        for p in model.parameters():
            p.requires_grad = True
        print(f"Phase 2: full fine-tune, {phase2_epochs} epoch(s), lr=1e-4", flush=True)
        best_acc = train_epochs(model, train_loader, test_loader,
                                phase2_epochs, 1e-4, "phase2", best_acc, ckpt_dir)

    summary = {"seed": SEED, "params": n_params, "best_test_acc": round(best_acc, 4),
               "smoke_test": args.smoke_test}
    with open(ckpt_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE. Best test accuracy: {best_acc:.4f} "
          f"(this is the Week 2 baseline accuracy)", flush=True)


if __name__ == "__main__":
    main()
