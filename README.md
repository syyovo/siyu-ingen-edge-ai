# siyu-ingen-edge-ai

Edge AI & Model Compression Internship — inGen Dynamics

Compressing deep learning models with quantization, pruning, and knowledge
distillation so they can run on resource-constrained edge devices, benchmarked
against per-platform deployment targets (model size and inference latency).

**Intern:** Siyu Xiang · Georgia Institute of Technology (Computer Engineering)
**Supervisor:** Iqbal Patel · inGen Dynamics
**Program:** 4-week remote internship

---

## Project overview

inGen Dynamics builds *physical intelligence* — AI that perceives, reasons, and
acts on real robots (Aido, Sentinel, Fari, and others), running **on-device**
rather than in the cloud. On-device inference means models must fit within tight
memory and latency budgets on hardware without a GPU.

This project prototypes that compression workflow on a standard model and public
dataset. Each week builds on the last:

- **Week 1** — Edge AI landscape, hardware constraints, and baseline profiling
- **Week 2** — Post-training quantization and structured pruning
- **Week 3** — Knowledge distillation and a lightweight NAS sketch; compression playbook
- **Week 4** — Rover verification run, reproducibility pass, capstone report, deck, and retrospective

All experiments run **CPU-only** to simulate edge deployment. Latency is measured
with `timeit` at batch size = 1.

---

## Model & dataset

**Model:** MobileNetV3-Small (`torchvision.models.mobilenet_v3_small`, ImageNet-pretrained)
**Dataset:** CIFAR-10 (via HuggingFace `datasets`: `load_dataset('uoft-cs/cifar10')`)

Selection rationale:

- **Small enough for CPU** — MobileNetV3-Small is designed for edge/mobile
  hardware (2,542,856 parameters as loaded, ≈113 M FLOPs per inference), so it
  runs without a GPU and leaves clear room to study how far compression can push
  size and latency.
- **Public, easy-to-evaluate dataset** — CIFAR-10 is small, loads via
  HuggingFace `datasets`, and evaluates on CPU, making it a reliable testbed
  for measuring accuracy before vs. after compression.
- **Directly loadable in PyTorch** — pretrained model from torchvision, dataset
  via HuggingFace; no external checkpoints required.
- **Maps to a real use case** — image classification on a lightweight model
  mirrors the on-device perception task on platforms like Aido Rover.

### Baseline profile — two stages

- **As loaded (Week-1 profiling):** ImageNet-pretrained backbone with its
  1000-class head — 2,542,856 params · **10.31 MB** weight file · **40.46 ms**
  CPU latency (bs=1, mean of 100 runs, Apple M1 CPU).
  Details: `notebooks/W01_Baseline_Notebook.ipynb`.
- **Deployment baseline (the "before" side of every W2–W4 number):** after the
  10-class head swap + fine-tuning (`scripts/finetune_baseline.py`; checkpoint
  `checkpoints/baseline_best.pth`) — 1,528,106 params · **93.38%** CIFAR-10 test
  accuracy (full 10k split) · **6.24 MB** · 37.3–39.2 ms CPU latency across the
  four clean sessions committed here (37.29 ms 1-epoch pruning probe · 37.33 ms
  W04 quantization re-run · 38.5 ms W02 quantization memo session · 39.21 ms
  zero-shot pruning sweep — a 5.1% spread). The ft3 sweep session's latency
  column is load-contaminated from the 20% ratio onward and is superseded by
  architecture-identical clean measurements (its accuracies and sizes are
  seed-deterministic and unaffected); latencies taken under known background
  load are flagged wherever quoted. See the capstone report's Sections 3.4
  and 6.1.

---

## Repository structure

```
siyu-ingen-edge-ai/
├── notebooks/      # experiment notebooks (baseline, quantization, pruning, distillation, NAS)
├── scripts/        # training, sweep, verification, and figure-export scripts
├── experiments/    # sweep results (CSV), exported figures (PNG), raw run logs
├── checkpoints/    # model checkpoints (fine-tuned baseline + W04 verification runs)
├── logs/           # long-run training logs
├── reports/        # landscape briefing (W01), compression memo (W02), playbook (W03)
├── weekly/         # weekly Edge logs
├── capstone/       # capstone report, deck, retrospective (Week 4)
├── requirements.txt
└── README.md
```

---

## File guide

**notebooks/**

- `W01_Baseline_Notebook.ipynb` — profiles the as-loaded model end-to-end (params, FLOPs, size, CPU latency) and quantifies the Aido Rover deployment gap.
- `W02_Quantization_Notebook.ipynb` — FX-mode INT8 PTQ (per-channel / per-tensor), weight-only control, and the activation-range diagnosis of the INT8 collapse.
- `W02_Pruning_Notebook.ipynb` — structured L1 pruning: zero-shot 9-ratio sweep, 1-epoch recovery probe, and PELT / Dynp changepoint analysis on the ft3 curve.
- `W03_Distillation_Notebook.ipynb` — teacher-student distillation, T × α sweep, comparison table against PTQ / pruning / from-scratch alternatives.
- `W03_NAS_Sketch.ipynb` — GA micro-NAS with the orienteering-solver component mapping; best architecture vs hand-designed baseline.

**scripts/**

- `finetune_baseline.py` — 10-class head swap + fine-tuning; produces the 93.38% deployment baseline (`checkpoints/baseline_best.pth`).
- `prune_ft_sweep.py` — nine-ratio pruning sweep with 3 ep × 20k recovery fine-tuning → `experiments/W02_pruning_sweep_ft3.csv`.
- `qat_train.py` — 3-epoch quantization-aware training run (14.67%; does not repair the INT8 collapse).
- `w03_scratch_baseline.py` — from-scratch student control (47.40%) for the distillation comparison.
- `w03_distill_sweep.py` — T × α distillation sweep → `experiments/W03_distill_sweep.csv`.
- `w03_nas_ga.py` — GA micro-NAS driver (pop 10, 15 generations) → `experiments/W03_nas_ga_log.csv`. (Figure 7, `W03_nas_evolution.png`, is exported from this log by `notebooks/W03_NAS_Sketch.ipynb`.)
- `w04_rover_verification.py` — W04 verification of the 12% / 15% pruning extrapolation, identical ft3 protocol → `experiments/W04_rover_verification.csv`.
- `w04_fig3_actranges.py` — regenerates Figure 3 (per-layer activation ranges, stem outlier) → `experiments/W02_activation_ranges.png`.
- `w04_fig5_pareto.py` — Figure 5, combined four-technique Pareto → `experiments/W04_pareto_combined.png`.
- `w04_fig6_heatmap.py` — Figure 6, distillation T × α heatmap → `experiments/W04_distill_heatmap.png`.
- `remeasure_latency.py` — latency re-measurement of the saved W04 Rover checkpoints with an in-session baseline control (run 2026-07-30; the control flagged the session as loaded — capstone Section 10.1).

**experiments/** — data

- `W02_ptq_results.csv` — quantization profiles for FP32, INT8 per-channel and INT8 per-tensor (Week-4 re-run session). QAT per-epoch metrics live in `logs/qat_0714.log` (`checkpoints/qat_summary.json` is the 1-epoch smoke-test record); the weight-only control is in the notebook output.
- `W02_pruning_sweep.csv` — zero-shot pruning sweep (no recovery).
- `W02_pruning_sweep_ft.csv` — intermediate recovery-probe sweep (superseded by ft3).
- `W02_pruning_sweep_ft3.csv` — final recovery sweep (3 ep × 20k); source of the PELT changepoint finding. Latency column load-contaminated from 20% onward (superseded — report §3.4); accuracy/size authoritative.
- `W03_distill_sweep.csv` — final 8-epoch T × α sweep incl. the from-scratch control row.
- `W03_distill_sweep_3ep_underfit.csv` — earlier 3-epoch sweep, kept as an underfitting record.
- `W03_teacher_logits_5k.pt` — cached teacher logits (5k subset) for distillation training.
- `W03_nas_ga_log.csv` / `W03_nas_smoke_log.csv` — GA full-run and smoke-test logs.
- `W04_rover_verification.csv` — measured 12% / 15% verification rows (88.45% / 4.87 MB · 85.32% / 4.56 MB).

**experiments/** — figures

- `W01_deployment_gap.png` — Figure 1: baseline vs Aido Rover budget (size & latency bars).
- `W02_pareto_combined.png` — Figure 2: Week-2 combined accuracy-vs-size Pareto.
- `W02_activation_ranges.png` — Figure 3: top-10 activation ranges, 133× stem outlier.
- `W02_pruning_curve.png` — zero-shot dual-axis curve carrying the documented dead-end PELT-on-zero-shot annotation (changepoint 0.5 on a meaningless curve; superseded by the ft3 analysis, report §5.2; kept as a record, not cited by the report).
- `W02_pruning_dualaxis_ft3.png` — Figure 4: ft3 accuracy + size vs ratio, PELT changepoint marked.
- `W04_pareto_combined.png` — Figure 5: four-technique Pareto with platform budget lines.
- `W04_distill_heatmap.png` — Figure 6: T × α heatmap vs the 47.40% control.
- `W03_nas_evolution.png` — Figure 7: GA best fitness & size per generation (two jumps, at generations 4 and 11, both on the 4th conv gene).

**experiments/** — raw logs: `w03_sweep_log.txt`, `w03_scratch_log.txt`,
`w03_nas_full_log.txt`, `w03_nas_smoke_log.txt` (console output of the W3 runs).

**checkpoints/**

- `baseline_best.pth` — the fine-tuned 93.38% / 6.24 MB deployment baseline; the "before" side of every W2–W4 number. Regenerate with `scripts/finetune_baseline.py` if missing.
- `qat_summary.json` — 1-epoch QAT smoke-test record (seed 42, converted INT8 11.6%); archives the conversion-side reading in capstone Section 4.3, step 5. The 3-epoch run's metrics live in `logs/qat_0714.log`.
- `rover_prune12_ft3.pt` / `rover_prune15_ft3.pt` — the two W04 Rover verification models (88.45% / 4.87 MB and 85.32% / 4.56 MB).

**logs/** — long-run console output, kept because these runs are too slow to reproduce casually.

- `finetune_0712.log` — baseline fine-tuning run (head swap + full-network fine-tune) that produced `checkpoints/baseline_best.pth`.
- `prune_sweep_0714.log` — the nine-ratio ft3 pruning sweep, 6.7 h total (per-ratio timings and per-epoch losses).
- `qat_0714.log` — the 3-epoch QAT run; per-epoch INT8 accuracies 12.27% / 13.12% / 14.67%.
- `rover_verify.log` — the W04 12% / 15% verification run under the identical ft3 protocol.

**reports/**

- `W01_EdgeAI_Landscape.md` — Edge AI landscape briefing + platform constraint definitions.
- `W02_Compression_Memo.md` — 3-page quantization + pruning memo (eight-element reporting).
- `W03_Compression_Playbook.md` — 4-page per-platform compression playbook (main research deliverable).

**weekly/** — `Wk-01-EdgeLog.md`, `Wk-02-EdgeLog.md`, `Wk-03-EdgeLog.md`,
`Wk-04-Final-EdgeLog.md` (one log per week: progress, blockers, findings).

**capstone/**

- `W04_Capstone_Report.docx` — 12–15-page technical compression report (submitted deliverable; figures embedded).
- `W04_Capstone_Report.md` — markdown source of the same report, plus Appendix A (working inventory and figure manifest, repository copy only).
- `W04_Capstone_Deck.pptx` — 10-slide executive deck (conclusion-style titles).
- `W04_Retrospective.md` — 1-page retrospective incl. the PELT / *J. Lake Sci.* changepoint connection.

**root** — `requirements.txt` (pinned CPU-only dependency set; see Environment & reproduction below).

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
`numpy.random.seed()` at the top (seed 42), and is written to run end-to-end
without manual intervention. If `checkpoints/baseline_best.pth` is missing,
regenerate it with `scripts/finetune_baseline.py` before running the W2–W3
notebooks. Accuracy and model size reproduce exactly across sessions; absolute
latencies drift a few percent between clean sessions on this shared machine
(37.3–39.2 ms baseline spread), so every latency in the reports is quoted with
its measurement session; the ft3 sweep's latency column is load-contaminated
and superseded (capstone report, Section 3.4).

---

## Compression playbook (summary)

Full playbook: [`reports/W03_Compression_Playbook.md`](reports/W03_Compression_Playbook.md)
— per-platform recommendations backed by the W1–W3 measurements
(eight-element reporting; latencies are Apple M CPU, session-sensitive).

**One row has since been superseded.** The playbook's Rover recommendation was
written in Week 3 with 12% as an *extrapolation* (~5.0 MB at ~88%). The Week-4
verification run measured it: **88.45% / 4.87 MB**
(`experiments/W04_rover_verification.csv`, capstone Section 5.5). The table
below carries the measured values; the playbook is kept as written — the
verification is recorded in the capstone (Section 5.5) and in this table — so
the extrapolation-then-measurement loop stays visible.
Note also that 10% pruning does **not** meet the Rover budget (5.08 MB > 5 MB).

| Platform | Budget (MB / ms) | Primary recommendation |
|---|---|---|
| Aido Rover | < 5 / < 100 | Light pruning at 12% + recovery fine-tuning (W04-verified: 88.45% / 4.87 MB) |
| Sentinel Prime AI | < 3 / < 200 | Distillation route (needs a production training budget) |
| Aido Humanoid | — / < 50 | Structured pruning ≤ 10%, with fine-tuning |
| Fari | < 20 / < 500 | Deploy FP32 as-is (constraints non-binding) |
| Senpai | < 20 / — | Extrapolated from Fari; re-test INT8 on embeddings |
