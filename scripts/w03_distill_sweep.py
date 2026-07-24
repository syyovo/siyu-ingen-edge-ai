# W03 distillation sweep: T x alpha grid (9 configs), 3 epochs each.
# Incremental CSV append after EACH config - safe to interrupt and resume.
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random, time, csv
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small



SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
ROOT = Path.home() / 'siyu-ingen-edge-ai'
EPOCHS = 8

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TFM = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                          transforms.Normalize(MEAN, STD)])

class HFCifar10(Dataset):
    def __init__(self, hf_split, tfm): self.ds, self.tfm = hf_split, tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]; return self.tfm(item['img']), item['label']

class DistillSet(Dataset):
    def __init__(self, hf_split, tfm, logits):
        self.ds, self.tfm, self.logits = hf_split, tfm, logits
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        return self.tfm(item['img']), item['label'], self.logits[i]

def build_student(width=0.5):
    m = mobilenet_v3_small(weights=None, width_mult=width)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 10)
    return m

def distill_loss(s_logits, t_logits, labels, T, alpha):
    ce = F.cross_entropy(s_logits, labels)
    kd = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                  F.softmax(t_logits / T, dim=1),
                  reduction='batchmean') * (T * T)
    return alpha * ce + (1 - alpha) * kd

def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
    return correct / total

def main():
    ds = load_dataset('uoft-cs/cifar10')
    train5k = ds['train'].shuffle(seed=SEED).select(range(5000))
    t_logits = torch.load(ROOT / 'experiments' / 'W03_teacher_logits_5k.pt')
    train_loader = DataLoader(DistillSet(train5k, TFM, t_logits),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(HFCifar10(ds['test'], TFM),
                             batch_size=128, shuffle=False)

    out_csv = ROOT / 'experiments' / 'W03_distill_sweep.csv'
    ckpt_dir = ROOT / 'checkpoints' / 'w03_students'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_csv.exists():  # resume support: skip configs already finished
        with open(out_csv) as f:
            done = {(r['T'], r['alpha']) for r in csv.DictReader(f)}
    else:
        with open(out_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['T', 'alpha', 'epochs', 'test_acc', 'train_min'])

    for T in [2, 4, 8]:
        for alpha in [0.3, 0.5, 0.7]:
            if (str(T), str(alpha)) in done:
                print(f'skip T={T} a={alpha} (already done)', flush=True)
                continue
            torch.manual_seed(SEED)  # same init per config for fair comparison
            model = build_student()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            t0 = time.time()
            for ep in range(EPOCHS):
                model.train()
                for x, y, tl in train_loader:
                    opt.zero_grad()
                    loss = distill_loss(model(x), tl, y, T, alpha)
                    loss.backward()
                    opt.step()
                print(f'  T={T} a={alpha} epoch {ep+1}/{EPOCHS} '
                      f'({(time.time()-t0)/60:.1f} min)', flush=True)
            acc = evaluate(model, test_loader)
            mins = (time.time() - t0) / 60
            with open(out_csv, 'a', newline='') as f:
                csv.writer(f).writerow([T, alpha, EPOCHS, f'{acc:.4f}', f'{mins:.1f}'])
            torch.save(model.state_dict(), ckpt_dir / f'student_T{T}_a{alpha}.pth')
            print(f'DONE T={T} a={alpha}: acc={acc:.4f}, {mins:.1f} min', flush=True)

if __name__ == '__main__':
    main()