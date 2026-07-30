"""W04 - Figure 6: distillation T x alpha heatmap.
Reads experiments/W03_distill_sweep.csv; rows with empty T/alpha (the
from-scratch control) are dropped before building the 3x3 grid. Cells
beating the control (47.40%) get an outline; best cell gets a star.
Run from repo root: python scripts/w04_fig6_heatmap.py
Output: experiments/W04_distill_heatmap.png (dpi 150)"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTROL = 47.40
OUT = Path('experiments/W04_distill_heatmap.png')

def main():
    df = pd.read_csv('experiments/W03_distill_sweep.csv')
    cols = {c.lower().strip(): c for c in df.columns}
    aliases = {'T': ['t', 'temperature', 'temp'],
               'alpha': ['alpha', 'a', 'alpha_ce'],
               'acc': ['test_acc', 'acc', 'accuracy', 'final_acc', 'student_acc']}
    pick = {}
    for canon, opts in aliases.items():
        hit = next((cols[o] for o in opts if o in cols), None)
        if hit is None:
            raise SystemExit(f'[fig6] no column for "{canon}". Columns: {list(df.columns)}')
        pick[canon] = hit
    d = df[[pick['T'], pick['alpha'], pick['acc']]].copy()
    d.columns = ['T', 'alpha', 'acc']
    if d['acc'].max() <= 1.0:
        d['acc'] = d['acc'] * 100.0
    dropped = d[d[['T', 'alpha']].isna().any(axis=1)]
    if len(dropped):
        print(f'[fig6] dropped {len(dropped)} non-sweep row(s):')
        print(dropped.to_string(index=False))
    d = d.dropna(subset=['T', 'alpha'])
    Ts = sorted(d['T'].unique()); alphas = sorted(d['alpha'].unique())
    grid = np.full((len(Ts), len(alphas)), np.nan)
    for _, r in d.iterrows():
        grid[Ts.index(r['T']), alphas.index(r['alpha'])] = r['acc']
    if np.isnan(grid).any():
        raise SystemExit(f'[fig6] grid incomplete:\n{grid}')
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(grid, cmap='YlOrBr', aspect='auto')
    bi, bj = np.unravel_index(np.nanargmax(grid), grid.shape)
    for i in range(len(Ts)):
        for j in range(len(alphas)):
            v = grid[i, j]
            ax.text(j, i, f'{v:.2f}%' + (' *' if (i, j) == (bi, bj) else ''),
                    ha='center', va='center', fontsize=11,
                    fontweight='bold' if (i, j) == (bi, bj) else 'normal', color='black')
            if v > CONTROL:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor='black', lw=1.8))
    ax.set_xticks(range(len(alphas)), [f'alpha = {a}' for a in alphas])
    ax.set_yticks(range(len(Ts)), [f'T = {int(t)}' for t in Ts])
    ax.set_title('Distillation sweep: student test accuracy (8 ep x 5k, seed 42)\n'
                 f'outlined cells beat the from-scratch control ({CONTROL:.2f}%) - 2 of 9 do',
                 fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('test accuracy (%)')
    fig.tight_layout(); OUT.parent.mkdir(exist_ok=True); fig.savefig(OUT, dpi=150)
    print(f'best cell: T={int(Ts[bi])}, alpha={alphas[bj]} -> {grid[bi, bj]:.2f}% -> {OUT}')

if __name__ == '__main__':
    main()
