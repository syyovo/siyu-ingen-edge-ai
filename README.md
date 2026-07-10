# siyu-ingen-edge-ai

Edge AI & Model Compression Internship — InGen Dynamics

Compressing deep learning models with quantization, pruning, and knowledge
distillation so they can run on resource-constrained edge devices, benchmarked
against per-platform deployment targets (model size and inference latency).

**Intern:** Siyu Xiang · Georgia Institute of Technology (Computer Engineering)
**Supervisor:** Iqbal Patel · InGen Dynamics
**Program:** 4-week remote internship

---

## Project overview

InGen Dynamics builds *physical intelligence* — AI that perceives, reasons, and
acts on real robots (Aido, Sentinel, Fari, and others), running **on-device**
rather than in the cloud. On-device inference means models must fit within tight
memory and latency budgets on hardware without a GPU.

This project prototypes that compression workflow on a standard model and public
dataset. Each week builds on the last:

- **Week 1** — Edge AI landscape, hardware constraints, and baseline profiling
- **Week 2** — Post-training quantization and structured pruning
- **Week 3** — Knowledge distillation and a lightweight NAS sketch; compression playbook
- **Week 4** — Capstone report, deck, and retrospective

All experiments run **CPU-only** to simulate edge deployment. Latency is measured
with `timeit` at batch size = 1.

---

## Model & dataset

**Model:** MobileNetV3-Small (`torchvision.models.mobilenet_v3_small`, ImageNet-pretrained)
**Dataset:** CIFAR-10 (via HuggingFace `datasets`: `load_dataset('uoft-cs/cifar10')`)

Selection rationale:

- **Small enough for CPU** — MobileNetV3-Small is designed for edge/mobile
  hardware (2,542,856 parameters, ≈113 M FLOPs per inference), so it runs
  without a GPU and leaves clear room to study how far compression can push
  size and latency.
- **Public, easy-to-evaluate dataset** — CIFAR-10 is small, loads via
  HuggingFace `datasets`, and evaluates on CPU, making it a reliable testbed
  for measuring accuracy before vs. after compression.
- **Directly loadable in PyTorch** — pretrained model from torchvision, dataset
  via HuggingFace; no external checkpoints required.
- **Maps to a real use case** — image classification on a lightweight model
  mirrors the on-device perception task on platforms like Aido Rover.

Baseline profile (FP32): **10.31 MB** weight file · **40.46 ms** CPU latency
(bs=1, mean of 100 runs, Apple M-series CPU).
Details: `notebooks/W01_Baseline_Notebook.ipynb`.

---

## Repository structure

```
siyu-ingen-edge-ai/
├── notebooks/      # experiment notebooks (baseline, quantization, pruning, distillation, NAS)
├── experiments/    # precomputed compression-sweep results (CSV)
├── reports/        # landscape briefing (W01), compression memo (W02), playbook (W03)
├── weekly/         # weekly Edge logs (Wk-01-EdgeLog.md, ...)
├── capstone/       # capstone report, deck, retrospective (Week 4)
├── requirements.txt
└── README.md
```

---

## Environment & reproduction

**Requirements:** Python 3.13, CPU-only PyTorch.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the toolchain
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Verified toolchain: torch 2.12.1, torchvision 0.27.1, CUDA available: False
(CPU-only, as required for edge simulation).

For reproducibility, every notebook sets `torch.manual_seed()` and
`numpy.random.seed()` at the top, and is written to run end-to-end without manual
intervention.

---

## Compression playbook (summary)

*To be added in Week 3 — platform-specific recommendations for which compression
techniques to apply, at what settings, for each deployment target.*