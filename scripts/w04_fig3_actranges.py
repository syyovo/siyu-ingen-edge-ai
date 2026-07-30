"""
w04_fig3_actranges.py (v2) -- Figure 3: activation ranges recorded by INT8
calibration observers. v2 fixes: duplicate producer-op names no longer merge
bars (numeric y positions), '<built-in function add>' prettified to
'residual add', median label moved off the tick labels, stem annotation
placed inside its own bar.

Run from repo root:   python scripts/w04_fig3_actranges.py
Output:               experiments/W02_activation_ranges.png (dpi 150)
"""

import copy
import random
import warnings
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
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.quantized.engine = 'qnnpack'
warnings.filterwarnings('ignore', message='.*torch.ao.quantization is deprecated.*')
warnings.filterwarnings('ignore', message='.*FixedQParamsObserver.*')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TFM = transforms.Compose([
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


ds = load_dataset('uoft-cs/cifar10')
calib_hf = ds['train'].shuffle(seed=SEED).select(range(512))
calib_loader = DataLoader(HFCifar10(calib_hf, TFM), batch_size=32,
                          shuffle=False, num_workers=0)
print('calibration images:', len(calib_hf))

model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(1024, 10)
model.load_state_dict(torch.load('checkpoints/baseline_best.pth',
                                 map_location='cpu'))
model.eval()

per_channel_qconfig = tq.QConfig(
    activation=tq.HistogramObserver.with_args(reduce_range=False),
    weight=tq.PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
)
qmap = tq.QConfigMapping().set_global(per_channel_qconfig)
example_inputs = (torch.randn(1, 3, 224, 224),)

prepared = quantize_fx.prepare_fx(copy.deepcopy(model), qmap, example_inputs)
with torch.no_grad():
    for x, _ in calib_loader:
        prepared(x)
print('calibration done, extracting observer ranges...')

producer = {}
for node in prepared.graph.nodes:
    if node.op == 'call_module' and 'activation_post_process' in str(node.target):
        src = node.args[0]
        producer[str(node.target)] = str(getattr(src, 'target', src))


def pretty(op: str) -> str:
    if 'built-in function add' in op:
        return 'residual add'
    return op


ranges = []
for name, mod in prepared.named_modules():
    if 'activation_post_process' in name and hasattr(mod, 'min_val'):
        lo, hi = mod.min_val, mod.max_val
        if lo.numel() == 0:
            continue
        span = (hi.max() - lo.min()).item()
        ranges.append((pretty(producer.get(name, name)), span))

ranges.sort(key=lambda r: -r[1])
spans = [r[1] for r in ranges]
med = float(np.median(spans))
stem_span = max(s for op, s in ranges if op == 'features.0.0')
ratio = stem_span / med
print(f'observers: {len(ranges)} | median range: {med:.2f} | '
      f'stem (features.0.0): {stem_span:.2f} = {ratio:.1f}x median')
check = 'PASS' if round(ratio) == 133 else 'CHECK -- deck/report say 133x'
print(f'locked-number check (133x): computed {ratio:.1f}x -> {check}')

# --- plot: numeric y positions -> duplicate op names can no longer merge ---
CHARCOAL, ORANGE = '#22303A', '#F5871F'
top = ranges[:10][::-1]                    # largest ends up as the top bar
n = len(top)
ypos = np.arange(n)
vals = [s for _, s in top]
labels = [op for op, _ in top]
colors = [ORANGE if op == 'features.0.0' else CHARCOAL for op, _ in top]

fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.barh(ypos, vals, color=colors, height=0.62)
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xscale('log')
ax.set_xlim(left=min(vals) * 0.5, right=max(vals) * 6)

ax.axvline(med, color=ORANGE, ls='--', lw=1.2)
ax.annotate(f'median of all {len(ranges)} observers = {med:.1f}',
            xy=(med, n - 0.55), xytext=(med * 1.5, n - 0.55),
            color=ORANGE, fontsize=8, va='center',
            arrowprops=dict(arrowstyle='-', color=ORANGE, lw=0.8))

for y, (op, s) in zip(ypos, top):
    if op == 'features.0.0':
        ax.text(s * 0.92, y, f'{s:,.0f}  ({ratio:.0f}x median)',
                va='center', ha='right', fontsize=9, fontweight='bold',
                color='white')
    else:
        ax.text(s * 1.12, y, f'{s:,.0f}', va='center', fontsize=8.5,
                color=CHARCOAL)

ax.set_xlabel('activation range recorded by calibration observer (log scale)')
ax.set_title('Top-10 activation ranges: the stem conv breaks the INT8 grid',
             fontsize=11.5, color=CHARCOAL, loc='left', pad=12)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()

Path('experiments').mkdir(exist_ok=True)
OUT = 'experiments/W02_activation_ranges.png'
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print('saved:', OUT)