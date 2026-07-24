# W03 from-scratch control: same student, same data, same 8-epoch budget,
# plain cross-entropy only (no distillation). This isolates the KD effect.
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

def build_student(width=0.5):
    m = mobilenet_v3_small(weights=None, width_mult=width)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 10)
    return m

def main():
    ds = load_dataset('uoft-cs/cifar10')
    train5k = ds['train'].shuffle(seed=SEED).select(range(5000))
    train_loader = DataLoader(HFCifar10(train5k, TFM), batch_size=64, shuffle=True)
    test_loader = DataLoader(HFCifar10(ds['test'], TFM), batch_size=128, shuffle=False)

    torch.manual_seed(SEED)   # same init as every sweep config
    model = build_student()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f'  scratch epoch {ep+1}/{EPOCHS} ({(time.time()-t0)/60:.1f} min)',
              flush=True)
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
    acc = correct / total
    torch.save(model.state_dict(),
               ROOT / 'checkpoints' / 'w03_students' / 'student_scratch.pth')
    with open(ROOT / 'experiments' / 'W03_distill_sweep.csv', 'a', newline='') as f:
        csv.writer(f).writerow(['scratch', 'n/a', EPOCHS, f'{acc:.4f}',
                                f'{(time.time()-t0)/60:.1f}'])
    print(f'DONE from-scratch: acc={acc:.4f}', flush=True)

if __name__ == '__main__':
    main()