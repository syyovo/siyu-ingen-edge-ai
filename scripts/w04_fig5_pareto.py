"""W04 - Figure 5 (v2): combined four-technique Pareto front.
Accuracy/size from the ft3 CSV; pruning-chain LATENCIES from the clean
zero-shot session CSV (the ft3 latency column is load-contaminated and
superseded - report Section 3.4); quant/KD/control/baseline constants
from the report tables (KD/control 25.82/25.81 ms, baseline 39.21 ms). W04
12%/15% points drawn uncolored (white fill, crimson ring): their
latencies are loaded-session and must not join the clean color scale.
Run from repo root: python scripts/w04_fig5_pareto.py
Output: experiments/W04_pareto_combined.png (dpi 150)"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path('experiments/W04_pareto_combined.png')
BASELINE = dict(acc=93.38, size=6.24, lat=39.21)
QUANT = [dict(label='INT8 per-channel', acc=15.91, size=1.83, lat=3.42),
         dict(label='INT8 per-tensor', acc=10.00, size=1.65, lat=3.36),
         dict(label='QAT INT8 (3 ep)', acc=14.67, size=1.83, lat=3.42)]
KD = dict(acc=50.42, size=1.74, lat=25.82)
SCRATCH = dict(acc=47.40, size=1.74, lat=25.81)
PELT_RATIO = 0.40

def read_csv_normalized(path, need):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    aliases = {'ratio': ['ratio', 'nominal_ratio', 'prune_ratio'],
               'acc': ['test_acc', 'acc', 'accuracy', 'ft3_acc', 'final_acc'],
               'size': ['size_mb', 'size', 'model_size_mb'],
               'lat': ['latency_ms', 'lat_ms', 'latency']}
    out = {}
    for canon in need:
        hit = next((cols[a] for a in aliases[canon] if a in cols), None)
        if hit is None:
            raise SystemExit(f'[fig5] {path}: no column for "{canon}". Columns: {list(df.columns)}')
        out[canon] = df[hit]
    res = pd.DataFrame(out)
    if 'acc' in res.columns and res['acc'].max() <= 1.0:
        res['acc'] = res['acc'] * 100.0
    return res

def main():
    prune = read_csv_normalized('experiments/W02_pruning_sweep_ft3.csv',
                                ['ratio', 'acc', 'size'])
    prune = prune[prune['ratio'] > 0]
    zs = read_csv_normalized('experiments/W02_pruning_sweep.csv', ['ratio', 'lat'])
    prune = prune.merge(zs[zs['ratio'] > 0], on='ratio', validate='1:1')
    w04 = read_csv_normalized('experiments/W04_rover_verification.csv',
                              ['ratio', 'acc', 'size', 'lat'])
    w04 = w04[w04['ratio'] > 0]
    chain = (pd.concat([prune, w04], ignore_index=True)
             .sort_values('size', ascending=False).reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    lat_all = (list(prune['lat']) + [BASELINE['lat'], KD['lat'], SCRATCH['lat']]
               + [q['lat'] for q in QUANT])
    vmin, vmax = min(lat_all), max(lat_all)
    ax.plot([BASELINE['size']] + list(chain['size']),
            [BASELINE['acc']] + list(chain['acc']), '-', color='0.65', lw=1.2, zorder=1, label='pruning chain (all ratios)')
    sc = ax.scatter(prune['size'], prune['acc'], c=prune['lat'], cmap='viridis',
                    vmin=vmin, vmax=vmax, s=70, zorder=3, label='Structured pruning + ft3')
    ax.scatter(w04['size'], w04['acc'], facecolors='white', edgecolors='crimson',
               s=120, lw=1.8, zorder=4,
               label='W04 verification 12% / 15% (loaded-session lat., uncolored)')
    ax.scatter([BASELINE['size']], [BASELINE['acc']], c=[BASELINE['lat']], cmap='viridis',
               vmin=vmin, vmax=vmax, marker='*', s=380, edgecolors='k', lw=0.6,
               zorder=5, label='FP32 baseline')
    ax.scatter([q['size'] for q in QUANT], [q['acc'] for q in QUANT],
               c=[q['lat'] for q in QUANT], cmap='viridis', vmin=vmin, vmax=vmax,
               marker='s', s=90, edgecolors='k', lw=0.5, zorder=4, label='INT8 (PTQ / QAT)')
    ax.scatter([KD['size']], [KD['acc']], c=[KD['lat']], cmap='viridis', vmin=vmin,
               vmax=vmax, marker='^', s=130, edgecolors='k', lw=0.5, zorder=4,
               label='Distilled student')
    ax.scatter([SCRATCH['size']], [SCRATCH['acc']], c=[SCRATCH['lat']], cmap='viridis',
               vmin=vmin, vmax=vmax, marker='v', s=130, edgecolors='k', lw=0.5,
               zorder=4, label='From-scratch control')
    ax.axvline(5.0, color='tab:orange', ls='--', lw=1.2)
    ax.axvline(3.0, color='tab:red', ls='--', lw=1.2)
    ax.axhline(10.0, color='0.75', ls=':', lw=1.0)
    ax.text(5.0, 22, ' Rover < 5 MB', color='tab:orange', fontsize=9, rotation=90, va='bottom')
    ax.text(3.0, 22, ' Sentinel < 3 MB', color='tab:red', fontsize=9, rotation=90, va='bottom')
    ax.text(6.35, 11, 'random guess (10%)', color='0.55', fontsize=8, va='bottom', ha='right')
    ax.annotate('93.38%', (BASELINE['size'], BASELINE['acc']),
                textcoords='offset points', xytext=(-12, 8), fontsize=9)
    r12 = w04[abs(w04['ratio'] - 0.12) < 1e-6]
    if len(r12):
        ax.annotate('12%: 88.45% / 4.87 MB',
                    (float(r12['size'].iloc[0]), float(r12['acc'].iloc[0])),
                    textcoords='offset points', xytext=(8, 6), fontsize=9, color='crimson')
    pelt = chain[abs(chain['ratio'] - PELT_RATIO) < 1e-6]
    if len(pelt):
        ax.annotate('PELT 0.4', (float(pelt['size'].iloc[0]), float(pelt['acc'].iloc[0])),
                    textcoords='offset points', xytext=(8, -12), fontsize=9)
    ax.annotate('INT8: fast (~3.4 ms) but collapsed',
                (QUANT[0]['size'], QUANT[0]['acc']), textcoords='offset points',
                xytext=(10, 10), fontsize=8.5)
    ax.set_xlabel('Model size on disk (MB)')
    ax.set_ylabel('CIFAR-10 test accuracy (%)')
    ax.set_title('Compression Pareto front - four techniques, one protocol (seed 42, bs = 1)')
    ax.set_xlim(0, 6.9); ax.set_ylim(5, 100)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label('CPU latency (ms, clean-session color scale)')
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT, dpi=150)
    print(f'Plotted {len(prune)} pruning + {len(w04)} W04 + {len(QUANT)} INT8 + 3 other points -> {OUT}')
    print(chain.to_string(index=False))

if __name__ == '__main__':
    main()
