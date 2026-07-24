# W03 GA micro-NAS skeleton (B1).
# Methodology port of the orienteering-problem GA solver:
#   orienteering route (node sequence)   -> chromosome (architecture genes)
#   score maximized under time budget    -> accuracy maximized under size budget (< 5 MB, Aido Rover)
#   population init (random routes)      -> random architectures from discrete gene options
#   fitness (route score w/ feasibility) -> proxy-trained accuracy w/ size penalty
#   crossover (route segment swap)       -> one-point gene crossover
#   mutation (node swap)                 -> per-gene random re-draw (p = 0.2)
#   termination (generation budget)      -> fixed generation count
import torch, torch.nn as nn
import numpy as np, random, time, csv
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
ROOT = Path.home() / 'siyu-ingen-edge-ai'

SMOKE = False         # True: tiny run to verify code (B2). False: full run (B3).
POP_SIZE = 4 if SMOKE else 10
N_GEN    = 2 if SMOKE else 15
SIZE_BUDGET_MB = 5.0  # Aido Rover constraint

# --- Search space: genes are indices into these option lists ---
CH_OPTS = [8, 16, 24, 32, 48, 64]   # channels for each of 4 conv blocks
FC_OPTS = [64, 128, 256]            # hidden width of the classifier
N_GENES = 5                          # 4 conv genes + 1 fc gene

def random_chromosome():
    return [random.randrange(len(CH_OPTS)) for _ in range(4)] + \
           [random.randrange(len(FC_OPTS))]

def build_from_chromosome(chrom):
    chans, layers, in_c = [CH_OPTS[g] for g in chrom[:4]], [], 3
    for c in chans:
        layers += [nn.Conv2d(in_c, c, 3, padding=1), nn.BatchNorm2d(c),
                   nn.ReLU(), nn.MaxPool2d(2)]
        in_c = c
    fc = FC_OPTS[chrom[4]]
    layers += [nn.Flatten(), nn.Linear(in_c * 2 * 2, fc), nn.ReLU(),
               nn.Linear(fc, 10)]     # 32x32 input -> 2x2 after 4 pools
    return nn.Sequential(*layers)

def size_mb(model):
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6  # FP32 bytes

# --- Data: native 32x32 (fast proxy; NAS ranks architectures, it does not
#     produce final accuracies -- stated in the notebook) ---
TFM32 = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                                 [0.2470, 0.2435, 0.2616])])

class HFCifar10(Dataset):
    def __init__(self, hf_split, tfm): self.ds, self.tfm = hf_split, tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]; return self.tfm(item['img']), item['label']

def make_loaders():
    ds = load_dataset('uoft-cs/cifar10')
    tr = ds['train'].shuffle(seed=SEED).select(range(2000))
    te = ds['test'].select(range(1000))
    return (DataLoader(HFCifar10(tr, TFM32), batch_size=128, shuffle=True),
            DataLoader(HFCifar10(te, TFM32), batch_size=256, shuffle=False))

# --- Fitness: 1-epoch proxy training + size penalty; cached so duplicate
#     chromosomes across generations are never retrained ---
_cache = {}

def fitness(chrom, train_loader, test_loader):
    key = tuple(chrom)
    if key in _cache:
        return _cache[key]
    torch.manual_seed(SEED)
    model = build_from_chromosome(chrom)
    mb = size_mb(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for x, y in train_loader:
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
    acc = correct / total
    fit = acc - 10.0 * max(0.0, mb - SIZE_BUDGET_MB)   # hard-ish size penalty
    _cache[key] = (fit, acc, mb)
    return _cache[key]

# --- GA operators (orienteering-solver ports) ---
def tournament(pop, fits, k=3):
    idx = max(random.sample(range(len(pop)), k), key=lambda i: fits[i][0])
    return pop[idx][:]

def crossover(a, b):
    cut = random.randrange(1, N_GENES)
    return a[:cut] + b[cut:]

def mutate(chrom, p=0.2):
    for i in range(N_GENES):
        if random.random() < p:
            opts = CH_OPTS if i < 4 else FC_OPTS
            chrom[i] = random.randrange(len(opts))
    return chrom

def main():
    train_loader, test_loader = make_loaders()
    out_csv = ROOT / 'experiments' / 'W03_nas_ga_log.csv'
    if not out_csv.exists():
        with open(out_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['gen', 'chromosome', 'fitness',
                                    'proxy_acc', 'size_mb'])
    pop = [random_chromosome() for _ in range(POP_SIZE)]
    for gen in range(N_GEN):
        t0 = time.time()
        fits = [fitness(c, train_loader, test_loader) for c in pop]
        order = sorted(range(POP_SIZE), key=lambda i: fits[i][0], reverse=True)
        best = order[0]
        with open(out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            for i in order:
                w.writerow([gen, pop[i], f'{fits[i][0]:.4f}',
                            f'{fits[i][1]:.4f}', f'{fits[i][2]:.3f}'])
        print(f'gen {gen}: best fit={fits[best][0]:.4f} '
              f'acc={fits[best][1]:.4f} size={fits[best][2]:.3f}MB '
              f'chrom={pop[best]} ({(time.time()-t0)/60:.1f} min)', flush=True)
        elite = [pop[i][:] for i in order[:2]]          # elitism: keep top 2
        children = []
        while len(children) < POP_SIZE - len(elite):
            child = mutate(crossover(tournament(pop, fits),
                                     tournament(pop, fits)))
            children.append(child)
        pop = elite + children
    print('GA finished. Best cached architectures are in the CSV.')

if __name__ == '__main__':
    main()