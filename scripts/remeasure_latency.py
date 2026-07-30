"""Clean-session latency re-measurement for the two W04 Rover checkpoints.

Referenced by capstone report Section 10.1 ("clean re-measurement of the
two saved checkpoints"). Protocol is byte-identical to
scripts/w04_rover_verification.py: input randn(1,3,224,224), 10 warm-up
+ 100 timed runs, mean +- std, seed 42, default thread count.

HOW TO RUN (from repo root, ~3 min):
    1. Close every other app (clean session - this is the whole point).
    2. python scripts/remeasure_latency.py
    3. Paste the three RESULT lines back to the chat.

Expects:
    checkpoints/baseline_best.pth        (state_dict)
    checkpoints/rover_prune12_ft3.pt     (full model, saved by W04 verification)
    checkpoints/rover_prune15_ft3.pt     (full model, saved by W04 verification)
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_model():
    m = mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(1024, 10)
    return m


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


def load_or_die(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f'[remeasure] missing checkpoint: {p} - run from repo root.')
    return p


def main():
    print(f'[remeasure] torch {torch.__version__}, '
          f'threads {torch.get_num_threads()}, seed {SEED}, '
          f'protocol: 10 warmup + 100 timed, bs=1, 224x224')

    baseline = build_model()
    baseline.load_state_dict(
        torch.load(load_or_die('checkpoints/baseline_best.pth'),
                   map_location='cpu'))
    baseline.eval()

    models = [('baseline FP32', baseline)]
    for nn_pct in (12, 15):
        m = torch.load(load_or_die(f'checkpoints/rover_prune{nn_pct}_ft3.pt'),
                       map_location='cpu', weights_only=False)
        m.eval()
        models.append((f'prune {nn_pct}% + ft3', m))

    for name, m in models:
        mean, std = measure_latency_ms(m)
        print(f'RESULT  {name}: {mean:.2f} ms (std {std:.2f})')


if __name__ == '__main__':
    main()
