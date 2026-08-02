# Edge AI & Model Compression on inGen's Physical AI Platforms

**W04 Capstone Report — v1.0 (final review: 2026-08-03)**

**Author:** Siyu Xiang · Edge AI & Model Compression Intern
**Supervisor:** Iqbal Patel · inGen Dynamics
**Program:** 4-week remote internship · Repository: `syyovo/siyu-ingen-edge-ai` (v1.0)
**Scope:** Quantization, structured pruning, knowledge distillation, and a GA micro-NAS sketch on a MobileNetV3-Small / CIFAR-10 baseline, benchmarked against the deployment budgets of inGen's five physical AI platforms.

---

## 1. Executive Summary

inGen's platforms run AI on-device, on embedded CPUs without GPUs, so every model must fit a per-platform size budget and a per-platform latency budget. Over three experimental weeks plus a Week-4 verification run, this project benchmarked the three post-training compression families — quantization, structured pruning, and knowledge distillation — plus a genetic-algorithm architecture micro-search, all on one fine-tuned MobileNetV3-Small / CIFAR-10 baseline (93.38% accuracy / 6.24 MB / ~38 ms CPU latency in clean sessions), under a fixed measurement protocol (seed 42, CPU-only, batch size 1, 10 warm-up + 100 timed runs, full 10k test split). Three findings matter most for the PIC 2.0 deployment stack.

**Finding 1 — INT8 quantization fails on this backbone for a diagnosed reason, not a mysterious one.** PTQ INT8 delivers large size and speed gains — 3.4x (6.24 -> 1.83 MB) and ~11x (38.5 -> 3.4 ms; more than the ~4x either hardware mechanism predicts alone, Section 10.5) — but accuracy collapses to 10–16% under every tested configuration, and a 3-epoch QAT run does not repair it (14.67%, still rising when its budget ended — Section 4.4). A seven-step diagnostic chain isolates the cause: weight-only quantization is nearly lossless (90.64%), so the failure lives in activations — the stem output's range (~1986) is a 133x outlier against the network median (14.95), forcing a grid step (~7.8) that erases all normal-magnitude activations; FP32 fallback cannot escape it because the outlier tensor still crosses an INT8 boundary. The recommendation is architectural, not procedural: do not ship INT8 on this hard-swish/SE backbone; re-attempt it on a ReLU-based one, where the otherwise-ideal battery profile can be realized (INT4: not measurable in this toolchain; the grid-step argument extends the negative a fortiori — Section 4.5).

**Finding 2 — this compact architecture has no free pruning region under the tested recovery budget, and the PELT changepoint marks where the steep decline ends, not where damage begins or stops.** Across a nine-ratio structured L1 sweep with a 3-epoch x 20k recovery budget, even 10% pruning costs 3.9 pp (89.50%), and PELT (ruptures, RBF cost, stable across penalty in [0.5, 2], with Dynp agreeing on the same signal) places the changepoint at nominal ratio 0.4 (56.60%). By 0.4, coupling has physically removed 63% of parameters; beyond it the average decline halves (~9.2 -> ~4.6 pp per step) but another 22.9 pp still falls by 0.9. Operationally: 0.4 bounds the steep regime; the practical ceiling under this budget is ~10%, and every pruning step ships with a recovery fine-tuning budget.

**Finding 3 — the distillation signal is real (+3.02 pp over an identically-budgeted from-scratch control) but budget-limited, making "train small directly" the validated route for the tightest budgets rather than a deployable config today.** The best student (width x0.5, T=2 / alpha=0.5) reaches 50.42% / 1.74 MB / 25.8 ms versus 47.40% for the same architecture trained from scratch with identical data, epochs, and initialization. Only 2 of 9 (T, alpha) configurations beat the control — the nine-cell mean (45.97%) sits 1.43 pp *below* it, and each cell is a single seed-42 run — so the gain belongs to tuned KD, never to KD by default. Both post-hoc families fail through two distinct mechanisms with one shared origin — an efficiency-optimized backbone: quantization breaks on a single-tensor activation-range pathology; pruning finds no dormant capacity left to harvest. That shared origin is why the credible path to Sentinel Prime's < 3 MB is a smaller model trained into competence, not a large model cut down.

**Primary deployment recommendation.** Aido Rover: structured pruning at 12% with recovery fine-tuning, **measured in a Week-4 verification run at 88.45% / 4.87 MB** — inside the 5 MB budget with 0.13 MB headroom, within 0.5 pp of the Week-2 extrapolation. Fari (and by extrapolation Senpai): deploy FP32 as-is — constraints non-binding, accuracy decides. Aido Humanoid: pruning <= 10% + fine-tuning, latency-first. Sentinel Prime: nothing measured deploys today; the route is distillation at a production budget, with INT8 re-tested on a quantization-friendly backbone as secondary. Every recommendation carries a hardware qualifier (Section 9): INT8 acceleration and absolute latency depend on the target ISA.

---

## 2. Edge AI Deployment Context

### 2.1 Why inGen needs model compression

inGen Dynamics builds physical AI — robots whose models run on their own onboard hardware rather than in the cloud (PIC 2.0 / Origami platform documentation [1]): typically an ARM CPU with no GPU, limited memory, and often a battery. That creates the gap this project measures — models accurate enough to be useful are usually too large and too slow for the processors they must run on — and compression is the set of techniques that closes it.

### 2.2 Platform constraint table

Each platform's budget is derived from its sensor rate, power source, and interaction context (Week 1 landscape briefing, Section 2):

| Platform | Deployment context | Max model size | Max latency | Primary compression strategy |
|---|---|---|---|---|
| Aido Rover (outdoor patrol) | 10 Hz real-time sensor classification, Cortex-A-class CPU | < 5 MB | < 100 ms | Quantization → pruning + ft (after W2) |
| Sentinel Prime AI (security) | 5 Hz event classification, battery-powered | < 3 MB | < 200 ms | INT8 PTQ → distillation route (W2–W3) |
| Fari (elder companionship) | Companion dialogue, consumer ARM SoC | < 20 MB | < 500 ms | Distillation → FP32 as-is (constraints non-binding) |
| Aido Humanoid (bipedal research) | Multi-step action classification, 10 Hz+ control loop | — (latency-critical) | < 50 ms | Pruning ≤ 10% + ft (unchanged) |
| Senpai (education robot) | Course-retrieval embeddings | < 20 MB | — | Embedding quantization → re-test on target (no Senpai measurement) |

The "primary strategy" column records both the Week 1 plan and the post-experiment recommendation. Four of the five moved, but not in the same way: three were overturned by measurement (Rover, Sentinel Prime, Fari), and one (Senpai) was downgraded to a testing plan because no Senpai-specific experiment was run at all — an absence of evidence, not a contrary result. Only the Humanoid plan survived unchanged. The distance between the two entries is itself a result (Sections 4–6).

### 2.3 Hardware rationale for the budgets

**Why 10 Hz requires < 100 ms.** A 10 Hz sensor delivers a sample every 100 ms; inference must finish before the next one arrives or the robot reacts to where an obstacle *was*. The window also covers sensor reads, preprocessing, and decision logic, so the true model budget sits below the ceiling.

**Why INT8 should improve latency on ARM edge CPUs even without a GPU.** Three hardware mechanisms, none requiring a GPU: (1) SIMD — a 128-bit NEON register holds 16 INT8 values versus 4 FP32, so one instruction does 4x the multiply-accumulate work; (2) memory traffic — INT8 weights move 4x fewer bytes, the likelier bottleneck at batch size 1, where each weight is loaded from DRAM and used once; (3) integer ALU — integer multipliers are simpler circuits than floating-point units, costing less energy per operation, which matters on battery platforms like Sentinel Prime. Each mechanism alone caps the predicted speedup near 4x; the measured 11.3x and its attribution gap are discussed in Section 10.5. These speedups require wide integer SIMD (ARM NEON, RISC-V with V); every recommendation therefore states its hardware class (Sections 9–10).

### 2.4 Compression taxonomy used in this project

Following the four-category survey framing (Cheng et al. [2]): **quantization** (fewer bits; PTQ needs only calibration data, QAT retrains under simulated quantization), **structured pruning** (remove whole channels — the result stays a dense, fast model), **knowledge distillation** (train a small student against a teacher's softened outputs — the one family that can change the architecture), and **NAS** (search architectures under a hardware budget; MobileNetV3 itself is partly NAS-found [6], so the baseline already embodies this).

---

## 3. Baseline Model and Deployment Gap

### 3.1 Model and dataset selection

MobileNetV3-Small (torchvision, ImageNet-pretrained) on CIFAR-10 (HuggingFace `uoft-cs/cifar10`), per the Week-1 criteria: CPU-trainable, publicly available, importable as a PyTorch checkpoint, and a perception task mirroring Aido Rover's. One selection property became central later: the backbone is itself efficiency-optimized and partly NAS-designed — precisely why post-hoc compression found no slack in it (Sections 4–5).

### 3.2 As-loaded profile and the Week 1 deployment gap

| Metric | As-loaded FP32 (ImageNet head) | Aido Rover target | Status |
|---|---|---|---|
| Parameters | 2,542,856 | — | — |
| FLOPs (bs=1) | ~113 M (56.5 M MACs) | — | — |
| Weight file size | 10.31 MB | < 5 MB | 2.1x over budget |
| CPU latency (bs=1, mean of 100) | 40.46 ms | < 100 ms | Passes on dev machine — see caveat |

> **[FIGURE 1 — Baseline vs Aido Rover deployment gap (size and latency bar charts).** Source: `experiments/W01_deployment_gap.png` — already exported by `W01_Baseline_Notebook.ipynb`. Status: EXISTS.]

Size, not latency, is the blocker — with a caveat that recurs throughout this report: the development machine is an Apple M1 CPU, substantially faster than the Cortex-A-class target, so the latency margin cannot be trusted at 1:1 on real hardware.

### 3.3 Fine-tuned deployment baseline and the corrected gap

The as-loaded model carries a 1000-class ImageNet head that CIFAR-10 does not need. Fine-tuning (ImageNet weights -> 10-class head, 2 epochs head-only + 2 epochs full network, Adam) produced the deployment baseline used by every subsequent experiment:

**93.38% test accuracy / 6.24 MB / 1,528,106 parameters / 37.3–39.2 ms CPU latency across clean sessions (see Section 3.4).**

Swapping the 1000-class head for a 10-class head alone removed ~1M parameters, shrinking the file from 10.31 MB to 6.24 MB. The honest correction to Week 1: the true Aido Rover deployment gap is **1.24 MB (1.25x), not the 2.1x first reported** — the W01 figure profiled the wrong head for the task. Wherever "baseline" appears from Section 4 onward, it means this fine-tuned 93.38% / 6.24 MB model.

### 3.4 Measurement protocol (applies to every number in this report)

Seed 42 everywhere (`torch`, `numpy`, `random`); PyTorch 2.12.1 CPU build on Python 3.13, Apple M1 CPU (4 threads); accuracy on the full 10k CIFAR-10 test split unless marked as a 1k quick-screen; model size = the actual `state_dict` file on disk (captures batch-norm buffers and format overhead, not params x 4 bytes); latency = batch size 1, 10 untimed warm-up runs, mean of 100 timed runs. Latency session note: the baseline was re-measured in every experimental session. Across the **four clean sessions** committed to this repository it reads **37.29 ms** (1-epoch pruning probe, `experiments/W02_pruning_sweep_ft.csv`), **37.33 ms** (Week-4 quantization re-run, `experiments/W02_ptq_results.csv`), **38.5 ms** (W02 quantization memo session), and **39.21 ms** (zero-shot pruning sweep, `experiments/W02_pruning_sweep.csv`) — a 5.1% spread, +-3.0% about the 38.1 ms mean. The **ft3 sweep session's latency column is excluded from the clean set**: from the 20% ratio onward its readings run 32–75% above the two architecture-identical clean sessions (which agree with each other within 1.8% at every ratio), the curve is non-monotonic (20% reads slower than 10%), and the training wall-clock in `logs/prune_sweep_0714.log` slows at the same ratios (smaller models, longer blocks) — background load arrived mid-sweep. Every ft3 latency in this report is therefore superseded by architecture-identical clean measurements: latency depends only on the pruned graph, which recovery fine-tuning does not alter; ft3 accuracies and sizes are seed-deterministic and unaffected. The W04 verification session ran under known background load and is flagged wherever quoted. Accuracy and size reproduce exactly across sessions; latencies are quoted with their measurement session.

---

## 4. Quantization Results: PTQ INT8, QAT, and a Diagnosed Collapse

### 4.1 Setup

Deployment context: Sentinel Prime (< 3 MB, battery-first, always-on) is the platform whose Week-1 plan leaned hardest on INT8 (Section 2.2), so this section decides that plan. Static post-training quantization (PyTorch `torch.ao` FX graph mode, qnnpack backend), calibrated on 512 images drawn from the training split — never the test split. Week 1's expectation, following the MobileNetV3 paper's quantization-friendly design claims [6] and the Nagel et al. PTQ pipeline [3], was that per-channel INT8 would land within a few points of FP32. The measurement said otherwise, and the gap between expectation and outcome is the finding.

### 4.2 PTQ collapses under every configuration

INT8 delivers its size and speed gains in full — 6.24 -> 1.83 MB (3.4x) and 38.5 -> 3.42 ms (11.3x; larger than either Section-2.3 mechanism predicts alone, an attribution gap discussed in Section 10.5) — but accuracy collapses to near-random in all four tested configurations:

| Weight scheme | Activation observer | Test accuracy |
|---|---|---|
| per-channel | Histogram | 15.91% (full test set) |
| per-tensor | Histogram | 10.00% (full test set) |
| per-channel | MinMax | 12.2% (1k quick-screen) |
| official default qnnpack mapping | — | 10.2% (1k quick-screen) |

The direction matches Nagel et al. [3] — per-tensor < per-channel on depthwise-separable networks — but the magnitude exceeds it: even per-channel collapses.

> **[FIGURE 2 — Week-2 combined accuracy vs model size Pareto (FP32 baseline, INT8 per-channel / per-tensor / QAT, nine ft3 pruning points; latency as color; platform budget lines).** Source: `experiments/W02_pareto_combined.png` — exported by `W02_Quantization_Notebook.ipynb`. Note: the latency color scale in this frozen Week-2 figure includes ft3-session readings later found load-affected from the 20% ratio onward (Section 3.4); point positions (accuracy/size) are unaffected, and Figure 5 supersedes it with clean-session colors. Status: EXISTS.]

### 4.3 Root-cause diagnostic chain

Seven steps, each eliminating one hypothesis:

1. **Prediction distribution:** the INT8 model dumps ~78% of a test batch into 3 of 10 classes — systematic output distortion, not noise.
2. **Single-conv sanity check:** one Conv2d+ReLU quantized in isolation shows max abs error ~0.30 on outputs in [0, 4] — kernels compute correctly; toolchain bug ruled out.
3. **Observer swap (Histogram -> MinMax):** still collapsed — calibration-strategy hypothesis ruled out.
4. **Official default qconfig mapping:** still collapsed — misconfiguration hypothesis ruled out.
5. **QAT cross-check:** training loss falls steadily (1.90 -> 0.58) while the simulated (fake-quant) and converted true-INT8 accuracies agree — 11.8% vs 11.6% — so the failure exists at the simulation level, independent of the qnnpack backend. Provenance: the converted-INT8 side is archived in `checkpoints/qat_summary.json` (1-epoch smoke-test record, `best_int8_test_acc: 0.116`); the paired 11.8% fake-quant reading survives only in the notebook's narrative — the committed `logs/qat_0714.log` predates the per-epoch simulated-accuracy print now in `scripts/qat_train.py`. The step establishes the *agreement* of the two evaluation paths, not an accuracy level; the full run's per-epoch converted accuracies are 12.27% / 13.12% / 14.67% (`logs/qat_0714.log`).
6. **Experiment A — weight-only quantization: 90.64%** on the full test set (−2.74 pp vs baseline). Weight quantization is essentially lossless; **the collapse is caused by activation quantization.**
7. **Experiment B — per-layer activation ranges:** the stem convolution output (`features.0.0`) spans −1073.9 … 912.2 (range ~1986) against a network-wide median of 14.95 — a **133x outlier**. Its INT8 grid step is ~7.8, so all normal-magnitude activations (|x| <~ 4) collapse into a handful of bins; the wrecked stem output propagates network-wide, explaining the class-collapse of step 1. The observer table records the stem twice, and the distinction matters for the fix attempt below: `features.0.0` (the convolution output, range 1986.15) and `features.0.2` (its post-hard-swish output, −0.38 … 912.24, range 912.61 — hard-swish floors the negative side at −0.375). Secondary outliers (range 65–118, 4–8x the median) span `features.1–.7` and `features.12`.

> **[FIGURE 3 — Per-layer activation range profile, stem outlier annotated (log scale).** Source: `experiments/W02_activation_ranges.png` — generated by `scripts/w04_fig3_actranges.py` (same seed-42, 512-image calibration protocol as the notebook's Experiment B). Status: EXISTS.]

**Fix attempt — selective FP32 fallback: failed, and the failure is informative.** Keeping the stem block (`features.0`, both top outliers) in FP32 does not recover accuracy (14.7%, 1k quick-screen; module inspection confirms the fallback took effect). The pathology lives in *tensors*, not ops: the stem's output must still be quantized at the boundary into the next INT8 block — the post-hard-swish tensor of step 7 (range 912.61), whose grid step (~3.6) is still coarser than the entire normal activation scale (|x| <~ 4). Escaping it would mean extending FP32 through `features.1–.7`, forfeiting most of the compression benefit. Closed as a genuine negative result under a pre-committed time-box.

**Likely origin of the outliers.** After only 4 epochs of fine-tuning, the stem retains near-ImageNet weights while inputs changed to CIFAR-10 upscaled 32 -> 224; some channels respond with extreme values, and min/max calibration stretches the grid over them — the outlier failure mode Nagel et al. [3] describe for efficient architectures, amplified here by hard-swish/SE.

### 4.4 QAT: a budget-limited negative result

QAT (per-channel FakeQuantize, lr 1e-4, 3 epochs from the 93.38% baseline; `scripts/qat_train.py`, per-epoch metrics in `logs/qat_0714.log`) did not recover INT8 accuracy within its budget: 12.27% -> 13.12% -> 14.67% per epoch, still rising — and accelerating (+0.85, then +1.55 pp) — when the 3-epoch budget ended, with training loss still falling. By the standard applied to pruning (Section 5.3) and distillation (Section 7.2), that is a budget-limited negative, not proof that no QAT budget can repair the pathology (a larger run was not attempted; one epoch costs ~74 min). What it establishes, with step 5, is that the failure lives at the simulation level and is not fixed by a small-budget weight adaptation.

### 4.5 INT4

Not attempted: no CPU INT4 inference path exists in this toolchain on macOS (bitsandbytes is CUDA-oriented). Mechanistically, a ~17x coarser grid (255 quantization steps -> 15) can only worsen the same activation-range pathology, so the negative conclusion extends a fortiori. This is stated as a toolchain-plus-mechanism argument, not a measurement.

---

## 5. Structured Pruning: Nine-Ratio Sweep, Recovery Budget, and the PELT Changepoint

### 5.1 Method

Structured (channel-level) L1-magnitude pruning with `torch-pruning`, which physically removes channels and traverses MobileNetV3's residual/SE dependency groups — chosen over `torch.nn.utils.prune`, which only masks weights and therefore changes neither file size nor latency, making the eight-element table unfillable. The final 10-class Linear head is excluded (its output dimension *is* the task). Each ratio prunes a fresh copy of the baseline (not cumulative), so rows are comparable. A 30% single trial validated dependency-graph traversal before the sweep.

### 5.2 Zero-shot: no free redundancy

Zero-shot accuracy collapses at the very first ratio (10%: 93.4% -> 15.0%) and the whole 10–90% curve sits at random (~10%). Two things are at work. First, the collapse itself is the expected behavior of zero-shot structured pruning — removing whole channels breaks co-adapted weights and invalidates BatchNorm statistics (the standard explanation, not separately isolated here) — which is why recovery fine-tuning is part of the technique. It is *not* by itself evidence about redundancy: 95% of the drop at 10% (74.5 of 78.3 pp) is recovered by three epochs of fine-tuning without adding a single channel back (Section 5.3), so the claim that this efficiency-optimized backbone carries little dormant capacity rests on the post-recovery residual (3.88 pp at 10%), not on the zero-shot curve. Second, **dependency-group coupling makes nominal ratios understate physical removal**: nominal 10% removes 18.8% of parameters, nominal 40% removes 63%, and nominal 90% removes 98.6% (1,528,106 -> 21,948 parameters), because residual and SE groups shrink together. Size and latency gains are nonetheless real (50% -> 1.72 MB; 90% -> 0.17 MB and 4.89 ms against the 39.21 ms baseline of this zero-shot session). Latency scales sub-linearly — 69.6x fewer parameters buys 8.0x — because channel pruning leaves the layer count unchanged, so width-independent per-op overheads set a floor (likely, not profiled).

A methodological dead end is documented rather than hidden: PELT on this zero-shot curve is meaningless — a curve that is one long floor at random accuracy has no regimes to separate — which is what motivated the recovery-budget rerun below.

### 5.3 Recovery-budget experiment

A 1-epoch x 5k probe lifts 10%-pruning only to 55.0% and 20%+ not at all: that curve measures the recovery *budget*, not the architecture (loss was still falling when the epoch ended). The authoritative sweep therefore reran all nine ratios with a 12x budget (3 epochs x 20k images, `scripts/prune_ft_sweep.py`, 6.7 h unattended — 401.6 min summed over the nine ratios, `logs/prune_sweep_0714.log`):

| Nominal ratio | Zero-shot | +1 ep x 5k | +3 ep x 20k (ft3) |
|---|---|---|---|
| 10% | 15.04% | 55.01% | **89.50%** |
| 12% | — | — | **88.45%** (W04 verification) |
| 15% | — | — | **85.32%** (W04 verification) |
| 20% | 11.23% | 12.83% | **78.11%** |
| 30% | 7.63% | 12.38% | **68.04%** |
| 40% | 9.96% | 9.71% | **56.60%** |
| 50% | 10.54% | 11.27% | **54.91%** |
| 60% | 9.57% | 10.10% | **44.88%** |
| 70% | 10.00% | 10.00% | **46.54%** |
| 80% | 10.00% | 10.00% | **40.86%** |
| 90% | 10.00% | 10.00% | **33.71%** |

Full nine-ratio data: `experiments/W02_pruning_sweep_ft3.csv`; the 12%/15% rows come from the Week-4 verification run (`experiments/W04_rover_verification.csv`) under the identical ft3 protocol (hence the dashes in the Week-2 columns). The 60 -> 70% uptick (+1.66 pp) is a single-run observation: with one run per ratio and no measured run-to-run variance, noise and a real effect cannot be distinguished. Training loss was still falling at epoch 3 in every ratio, so the absolute recovered accuracies likely understate what a longer budget would reach — likely, not proven, since no held-out curve was tracked during fine-tuning. The shape-level conclusions (changepoint location, rankings) are expected to be budget-robust; the absolutes are not. The headline result, stated under this budget: recovery is real at low ratios, but there is **no loss-free region at 3 ep x 20k** — even 10% costs 3.9 pp.

### 5.4 Critical compression ratio (PELT)

**Changepoint: nominal ratio 0.4, test accuracy 56.60% at the changepoint.** Criterion, stated in advance rather than read off the plot: PELT (`ruptures`, RBF cost) places the changepoint at 0.4 for every penalty in [0.5, 2]; at penalty >= 3 it reports no changepoint at all rather than a different one, and exact dynamic-programming segmentation (Dynp, n_bkps = 1) — the same RBF cost on the same ten points, forced to return exactly one breakpoint — selects 0.4 as well. This is an internal-consistency check, two search strategies agreeing under one cost function, not corroboration from outside the curve (Section 10.6); with only ten points it is also the strongest check available.

**Mechanism — reading the changepoint correctly.** The changepoint separates steep collapse (~9.2 pp per step before 0.4: 93.4 -> 56.6) from a slower but continuing decline (~4.6 pp per step on average after — including one post-changepoint step, 50 -> 60%, that falls 10.0 pp, as steep as anything before it). It marks where the *steep regime ends*, not where damage begins and not a floor: by nominal 0.4, dependency-group coupling has physically removed 63% of parameters — entire feature-detector groups are gone, creating an information bottleneck that a 3-epoch budget cannot rebuild — and a further 22.9 pp of accuracy and 96% of the remaining parameters are lost between 0.4 and 0.9. Critically, the region before the changepoint is not a safe zone: degradation begins at the first step. Unlike over-parameterized networks, this curve has no pre-changepoint plateau — 0.4 is an upper bound on useful pruning, never an operating point.

> **[FIGURE 4 — Dual-axis: ft3 accuracy + model size vs nominal ratio, PELT changepoint and random-guess line marked.** Source: `experiments/W02_pruning_dualaxis_ft3.png` — already exported by `W02_Pruning_Notebook.ipynb`. Status: EXISTS.]

### 5.5 The Aido Rover operating point — extrapolation verified by measurement

Week 2 ended one notch short of the Rover budget: 10% + ft3 = 89.50% / 5.08 MB / 33.8 ms (clean-session latency, Section 3.4), 0.08 MB over, with ~12% *extrapolated* to clear 5 MB at ~88%. A Week-4 verification run closed that gap under the identical ft3 protocol (torch-pruning L1, head excluded, 3 ep x 20k, seed 42; `scripts/w04_rover_verification.py`), including a baseline re-check that reproduced 93.38% exactly:

| Config | Test acc | Size | Params (actual removal) | Latency |
|---|---|---|---|---|
| Prune 12% + ft3 | **88.45%** | **4.87 MB** | 1,186,341 (−22.4%) | 46.62 ms (loaded session) |
| Prune 15% + ft3 | 85.32% | 4.56 MB | 1,109,583 (−27.4%) | 43.53 ms (loaded session) |

**The measurement confirms the Week-2 extrapolation — measured 88.45% vs predicted ~88% — with 0.13 MB of budget headroom.** The new points slot monotonically into the W02 curve (10% → 89.50, 12% → 88.45, 15% → 85.32, 20% → 78.11) with no anomalies; nominal 12% physically removes 22.4% of parameters (dependency coupling, Section 5.2). Two caveats carry over unchanged: training loss was still falling at epoch 3 for both new points (0.527 → 0.254 at 12%; 0.759 → 0.376 at 15%), so these accuracies likely understate longer-budget recovery (Section 5.3's caveat applies); and this session's latencies were taken under background load (the baseline re-check read 50.61 ms vs 37.3–39.2 ms in clean sessions). The < 100 ms budget conclusion is untouched — even loaded, 46.62 ms sits far inside. A pre-presentation re-measurement (`scripts/remeasure_latency.py`, 2026-07-30) read 43.79 / 42.20 ms for the 12% / 15% checkpoints, but its in-session baseline control read 49.61 ms against the 37.3–39.2 ms clean cluster, so that session does not qualify as clean either; both sessions' values stand as annotated records (Section 10.1).

---

## 6. Pareto Frontier and Per-Platform Feasibility

### 6.1 Consolidated eight-element results

Reference: fine-tuned FP32 baseline, 93.38% / 6.24 MB. Accuracy and size are session-independent and reproduce exactly; only latency is session-dependent, so every latency below carries its session: **(a)** W02 quantization memo session, baseline 38.5 ms; **(g)** zero-shot pruning sweep, baseline 39.21 ms (`experiments/W02_pruning_sweep.csv`) — quoted for all pruning rows: latency depends only on the pruned graph (unchanged by fine-tuning), and the ft3 session's latency column is load-contaminated and superseded (Section 3.4); **(e)** W04 verification session, run under background load, baseline re-check 50.61 ms; **(f)** W03 distillation session (`W03_Distillation_Notebook.ipynb`), which took no baseline re-check — its "before" cells quote the clean-session range, and the directly comparable pair is the student vs its control within the session. Only configurations with all eight elements measured appear here; the full sweeps live in `/experiments/`.

| Technique | Ratio / bit-width | Base acc | Comp. acc | Acc loss (pp) | Size before | Size after | Lat. before | Lat. after |
|---|---|---|---|---|---|---|---|---|
| PTQ INT8 per-channel | 8-bit (size /3.4) | 93.38% | 15.91% | −77.47 | 6.24 MB | 1.83 MB | 38.5 ms (a) | 3.42 ms (a) |
| PTQ INT8 per-tensor | 8-bit (size /3.8) | 93.38% | 10.00% | −83.38 | 6.24 MB | 1.65 MB | 38.5 ms (a) | 3.36 ms (a) |
| QAT INT8, 3 ep | 8-bit | 93.38% | 14.67% | −78.71 | 6.24 MB | 1.83 MB* | 38.5 ms (a) | 3.42 ms* (a) |
| Prune 10% + ft3 | channels −10% (params −18.8%) | 93.38% | 89.50% | −3.88 | 6.24 MB | 5.08 MB | 39.21 ms (g) | 33.77 ms (g) |
| Prune 12% + ft3 (W04 verification) | channels −12% (params −22.4%) | 93.38% | 88.45% | −4.93 | 6.24 MB | 4.87 MB | 50.61 ms (e) | 46.62 ms (e) |
| Prune 15% + ft3 (W04 verification) | channels −15% (params −27.4%) | 93.38% | 85.32% | −8.06 | 6.24 MB | 4.56 MB | 50.61 ms (e) | 43.53 ms (e) |
| Prune 40% + ft3 (PELT point) | channels −40% (params −63%) | 93.38% | 56.60% | −36.78 | 6.24 MB | 2.37 MB | 39.21 ms (g) | 22.70 ms (g) |
| Prune 50% + ft3 | channels −50% | 93.38% | 54.91% | −38.47 | 6.24 MB | 1.72 MB | 39.21 ms (g) | 19.27 ms (g) |
| Prune 90% + ft3 | channels −90% (params −98.6%) | 93.38% | 33.71% | −59.67 | 6.24 MB | 0.17 MB | 39.21 ms (g) | 4.89 ms (g) |
| KD student, T=2 / alpha=0.5 | width x0.5 (params −73%, size /3.6) | 93.38% | 50.42% | −42.96 | 6.24 MB | 1.74 MB | 37.3–39.2 ms (clean) | 25.82 ms (f) |
| Same student, from scratch | width x0.5 | 93.38% | 47.40% | −45.98 | 6.24 MB | 1.74 MB | 37.3–39.2 ms (clean) | 25.81 ms (f) |

\* QAT produces the same quantization scheme as per-channel PTQ; size and latency measured on the per-channel artifact and reused. The retired ft3 latency column sits 32–75% above the architecture-identical clean values quoted here (50%: 33.71 ms ft3 vs 19.27 / 19.37 ms clean) — Section 3.4. (e) The W04 session ran under background load; a 2026-07-30 re-measurement read 43.79 / 42.20 ms but failed its own baseline control (49.61 ms) and is likewise session-annotated (Section 10.1). Committed-CSV note: `experiments/W02_ptq_results.csv` holds the Week-4 re-run (FP32 37.33 -> 3.34 ms, 11.18x); the (a) rows quote the original memo session (38.5 -> 3.42 ms, 11.3x) — accuracy and size identical, milliseconds within normal clean-session drift. The GA micro-NAS best (0.465 MB, 32.7% *proxy*) is excluded: its proxy metric is not comparable to full-test accuracy (Section 8).

### 6.2 The frontier and the empty corner

On the accuracy-vs-size plane, the Pareto frontier runs from the FP32 baseline down the pruning-recovery chain (10% -> 20% -> … -> 90%, with 60% excepted — it is dominated by 70%, which is both smaller and more accurate); the distilled student sits below the pruned model of comparable size, and the INT8 points occupy the small-size / collapsed-accuracy corner — real compression, unusable accuracy. Latency as point color shows why INT8 stays strategically interesting despite the collapse: 3.3–3.4 ms, the fastest corner measured — the speed mechanism is intact where accuracy fails.

The decision-relevant feature is the **empty upper-left region**: no measured configuration achieves > 85% accuracy below 3 MB. That empty corner *is* the Sentinel Prime infeasibility finding, and it is why the Sentinel recommendation is a route (distillation at production budget) rather than a config.

> **[FIGURE 5 — Combined four-technique Pareto: accuracy vs size for baseline, INT8 (per-channel, per-tensor, QAT), all nine ft3 pruning points, KD student, from-scratch control; points colored by latency; platform budget lines at 5 MB and 3 MB.** Source: `experiments/W04_pareto_combined.png` — generated by `scripts/w04_fig5_pareto.py` from `experiments/W02_pruning_sweep_ft3.csv` (accuracy/size), `experiments/W02_pruning_sweep.csv` (clean-session latencies for the color scale), and `experiments/W04_rover_verification.csv` (12%/15% points, drawn uncolored); INT8, KD, control, and baseline values are constants taken from the report tables. The W2-only version exists as `experiments/W02_pareto_combined.png`. Status: EXISTS (v2 — clean-session latency color scale).]

### 6.3 Per-platform feasibility

| Platform | Budget (MB / ms) | Best measured option | Meets budget? |
|---|---|---|---|
| Aido Rover | < 5 / < 100 | Prune 12% + ft3: 88.45% / 4.87 MB / 46.6 ms (e) | **Yes** — 0.13 MB headroom; within 0.5 pp of the W02 extrapolation |
| Sentinel Prime AI | < 3 / < 200 | None viable: prune 40% fits size (2.37 MB) at 56.60%; INT8 fits size (1.83 MB) with collapsed accuracy | No — distillation route is the only credible path (Section 7, 9) |
| Aido Humanoid | — / < 50 | Prune 10% + ft3: 33.8 ms on dev machine (clean session) | Latency passes on dev hardware; must be re-validated on target ARM (Section 10) |
| Fari | < 20 / < 500 | FP32 baseline as-is: 93.38% / 6.24 MB / ~38 ms | Yes, with wide margin — constraints non-binding |
| Senpai | < 20 / — | Extrapolated from Fari (vision-CNN evidence only) | Testing plan, not sign-off (Section 9.5) |

---

## 7. Knowledge Distillation: Teacher–Student Scheme, T/alpha Sweep, and the Four-Way Comparison

### 7.1 Design rationale — capacity as the only variable

**Teacher:** the project's own fine-tuned baseline (93.38% / 6.24 MB; 37.3–39.2 ms across clean sessions, the latency reference here), rather than a larger public model, so the entire W1–W3 comparison chain stays anchored to one reference model. **Student:** MobileNetV3-Small at `width_mult=0.5` — identical layer structure, half the channels: 409,394 parameters (−73.2%; within the plan's 50–70% target). A same-family width-scaled student keeps capacity as the *only* variable between teacher and student — any gap is attributable to capacity plus training signal, never to architectural confounds — and stays directly comparable to the Week-2 pruned models.

**Loss (Hinton et al. [5]):** L = alpha * CE(student, labels) + (1 − alpha) * KL(softmax(student/T) || softmax(teacher/T)) * T^2, with teacher logits precomputed once over the 5k training subset (`experiments/W03_teacher_logits_5k.pt`) so the teacher never runs inside the training loop. **Sweep:** T in {2, 4, 8} x alpha in {0.3, 0.5, 0.7} — 9 configurations, 8 epochs x 5k images each, seed 42. **Control:** the identical student trained from scratch (hard labels only) with the same data, budget, and initialization seed, so the distillation signal is isolated.

### 7.2 A documented dead end that set the budget

The first sweep ran 3 epochs per configuration and returned 10.00% — random-guess — on its first config. A 60-step probe showed CE loss falling normally (2.15 -> 1.58), so the mechanics were sound and the failure was **budget starvation**, not a bug — the same lesson as the Week-2 recovery probe. The sweep was re-run at 8 epochs; the failed run is archived (`experiments/W03_distill_sweep_3ep_underfit.csv`) rather than deleted. The control uses the same 8-epoch budget.

### 7.3 Sweep results

| Test accuracy | alpha = 0.3 | alpha = 0.5 | alpha = 0.7 |
|---|---|---|---|
| **T = 2** | 48.38% | **50.42%** | 46.75% |
| **T = 4** | 46.41% | 43.67% | 46.37% |
| **T = 8** | 39.26% | 45.43% | 47.00% |

Best distilled student (T=2, alpha=0.5): **50.42% / 1.74 MB / 25.82 +- 0.77 ms**. From-scratch control: **47.40% / 1.74 MB / 25.81 +- 0.80 ms** — identical within noise, as they must be for two runs of the same architecture: the distillation signal is an accuracy effect, not a latency one. It is worth **+3.02 pp** under this controlled comparison — but only **2 of 9** configurations beat the control, and the worst (T=8, alpha=0.3 at 39.26%) lands ~8 pp *below* it. The nine-cell mean is 45.97%, 1.43 pp below the control, and each cell is a single seed-42 run with no measured run-to-run variance — the +3.02 pp belongs to the tuned configuration, not to distillation by default.

> **[FIGURE 6 — T x alpha heatmap of student test accuracy, from-scratch control level marked.** Source: `experiments/W04_distill_heatmap.png` — generated by `scripts/w04_fig6_heatmap.py` from `experiments/W03_distill_sweep.csv`. Status: EXISTS.]

### 7.4 Mechanism, and how distillation compares with quantization and pruning

**Why low temperature wins here.** The teacher is already confident (93.38%), so mild softening (T=2) transfers useful inter-class structure while preserving a strong training signal; T=8 over-smooths the teacher's distribution into a weak, near-uniform target — the 8 pp-below-control cell is what an over-smoothed signal costs. The gain is hyperparameter-sensitive enough that a deployment pipeline should treat (T, alpha) as a swept design decision, never a default.

**The inheritance gap.** The capacity-matched comparison in Section 6.1 is prune-50% vs the distilled student: 403,354 vs 409,394 parameters, 1.72 vs 1.74 MB — and the pruned model leads **54.91% to 50.42%**, a 4.49 pp gap for the route that keeps trained weights. Budgets remain unmatched (3 ep x 20k from the fine-tuned checkpoint, lr 1e-4, vs 8 ep x 5k from random init, lr 1e-3), so the gap is *consistent with* the inheritance mechanism — pruning keeps trained feature detectors; a student re-grows them — rather than proof of it; capacity is the only controlled variable. The strategic consequence stands either way: pruning wins wherever the budget permits keeping most of the trained network; distillation is the route for budgets *below* what light pruning can reach (Sentinel's < 3 MB), because a student sized to the target from the start avoids the post-hoc compression penalty entirely — provided it gets a production-scale training budget rather than the exploration budget used here.

---

## 8. GA Micro-NAS Sketch: Methodology, Orienteering-Solver Mapping, and Best Architecture

### 8.1 Methodology

A genetic-algorithm micro-search over a small CNN space, the Aido Rover size budget wired into the objective. **Chromosome:** 5 genes — 4 conv-channel genes over {8, 16, 24, 32, 48, 64} + 1 FC-width gene over {64, 128, 256}. **Fitness = proxy accuracy − 10 x max(0, size_MB − 5).** **Settings:** population 10, 15 generations, seed 42; tournament selection (k = 3), one-point crossover, per-gene mutation (p = 0.2), elitism (top-2), and a fitness cache. **Proxy protocol:** 1 epoch on 2,000 CIFAR-10 images at 32x32, evaluated on a 1,000-image subset — proxy accuracies rank candidates within the search and are *not* comparable to the full-training numbers elsewhere in this report. The search runs as a script (`scripts/w03_nas_ga.py`); the notebook analyzes the run log (`experiments/W03_nas_ga_log.csv`).

### 8.2 Orienteering-solver mapping

The search machinery is a direct port of my orienteering-problem GA solver — the same algorithmic skeleton with a different genome:

| GA component | Orienteering solver | Micro-NAS equivalent |
|---|---|---|
| Genome | Route encoding (node-sequence chromosome) | Architecture chromosome: 4 conv-channel genes + 1 FC-width gene |
| Fitness | Route score under a time/distance budget | Proxy accuracy − 10 x max(0, size_MB − 5): the Rover 5 MB budget as the "distance budget" |
| Selection | Favor high-scoring routes | Tournament selection, k = 3 |
| Crossover | Route segment exchange | One-point gene crossover |
| Mutation | Node swap / perturbation | Per-gene random re-draw, p = 0.2 |
| Addition beyond the solver | — | Elitism (top-2) + fitness cache to avoid retraining duplicates |

The transfer is the point: a budget-constrained combinatorial search is the same problem shape whether the genome is a route or a channel-width vector.

### 8.3 Results — genuine selection pressure, honest non-win

**Best architecture found: conv channels [64, 64, 48, 32] + FC 256 (chromosome [5, 5, 4, 3, 2]) — 0.465 MB at 32.7% proxy accuracy.** The evolution trace shows genuine selection pressure rather than drift: both fitness jumps (generations 4 and 11) occurred on the **same gene — the 4th conv block, growing from its generation-0 initial value of 8 channels to 24 and then 32**. The GA added capacity at the bottleneck while the other genes stayed at high-capacity options — the intervention a hand-designer would apply.

As the plan anticipated, the GA did not beat the hand-designed references — and the numbers are additionally non-comparable across training protocols, which is why the GA row is excluded from the Section 6.1 eight-element table:

| Model | Size | Accuracy | Training budget |
|---|---|---|---|
| GA-found best | 0.465 MB | 32.7% (proxy) | 1 ep x 2k, 32px |
| Hand-designed student (width-0.5 MNv3, from scratch) | 1.74 MB | 47.40% | 8 ep x 5k, 224px |
| Baseline (fine-tuned MNv3-Small teacher) | 6.24 MB | 93.38% | Full fine-tune |

**Two limitations, stated plainly.** (1) The < 5 MB size penalty **never activated**: the search space caps out well below the budget, so the hardware constraint was non-binding in this run — a tighter budget or a latency term would make the trade-off active — the first change for a follow-up search. (2) Proxy fidelity (1 ep, 32px) is assumed, not validated. The sketch's role is methodological — working search machinery with a hardware constraint in the fitness and a one-to-one solver mapping; a foundation, not a deployment path.

> **[FIGURE 7 — Best fitness and best-candidate size per generation (dual-axis), the two jumps at generations 4 and 11, both attributable to the 4th-conv-block gene.** Source: `experiments/W03_nas_evolution.png` — already exported by `W03_NAS_Sketch.ipynb`. Status: EXISTS.]

---

## 9. Compression Playbook: Per-Platform Recommendations

**Cross-cutting finding.** A compact, efficiency-optimized architecture resists post-hoc compression through two distinct mechanisms with one shared origin: INT8 collapses on a single-tensor activation-range pathology (hard-swish stem) that neither fallback quantization nor the 3-epoch QAT run repairs, and pruning shows no loss-free region under the tested recovery budget — the 3.88 pp post-recovery residual at 10% being the direct evidence that dormant capacity is scarce. The stronger routes are light pruning with a recovery budget, or training a smaller model directly (distillation — validated direction, budget-limited in Week 3). Every recommendation below states its hardware class: INT8 acceleration is ISA-dependent, and absolute latencies come from an Apple M1 dev CPU, not target silicon.

### 9.1 Aido Rover — outdoor patrol, onboard perception (< 5 MB, < 100 ms; ARM Cortex-A class)

**Primary: structured pruning at 12% + recovery fine-tuning — measured, not extrapolated: 88.45% / 4.87 MB / 46.6 ms (loaded session; a re-measurement read 43.8 ms but failed its baseline control — Section 10.1), 0.13 MB inside budget.** The Week-4 run landed within 0.5 pp of the Week-2 extrapolation; a 15% bracketing point (85.32% / 4.56 MB) prices deeper headroom at roughly 1 pp per nominal percent in this range. Quantization is excluded *on this baseline* (PTQ 15.91%, 3-ep QAT 14.67% — the activation pathology of Section 4; the speed mechanism itself is intact at 3.42 ms). **Ratio guidance:** 10–15% by accuracy-vs-headroom need (10%: 89.50% / 5.08 MB — over budget; 12%: 88.45% / 4.87 MB; 15%: 85.32% / 4.56 MB); PELT 0.4 marks the end of the steep regime, never a safe limit; every step ships with a recovery budget. **Secondary:** the distilled width-0.5 student (1.74 MB / 25.8 ms) if the size budget ever tightens well below 5 MB — validated route, currently budget-starved at 50.42%.

### 9.2 Sentinel Prime AI — battery-powered, always-on (< 3 MB, < 200 ms; ARM edge CPU)

**Honest finding first: no measured W2/W3 configuration deploys today.** Under 3 MB the options are pruning @ 40% (2.37 MB, 56.60%) and INT8 (1.83 MB, collapsed) — the budget-compliant pruning region (>= 40% by the size curve) lies at or past the PELT changepoint, inside the collapsed regime. **Primary route: knowledge distillation into a purpose-sized student at a production training budget.** The width-0.5 student fits with headroom (1.74 MB / 25.8 ms); the +3.02 pp best-cell result validates the signal (hyperparameter-sensitive — Section 7.3), but at the W3 budget it reaches only 50.42% — validated direction, insufficient budget; production-budget accuracy is TBD (90%-class is a target, not a promise). **Secondary:** re-attempt INT8 on a ReLU-based backbone — the 3.42 ms / 1.83 MB profile is otherwise ideal for a battery platform; a redesign recommendation, not a measured config.

### 9.3 Aido Humanoid — bipedal research, joint-sensor sequences (< 50 ms; ARM with NEON-class SIMD)

Latency is the binding constraint (no size budget defined). **Primary: structured channel pruning <= 10% with fine-tuning** — noting the W2 negative honestly: zero-shot pruning collapses at every ratio, and even with recovery 10% costs 3.9 pp (Section 5). 10% + ft3 = 33.8 ms on the dev machine (clean session), comfortably under 50 ms *there*; the margin must be re-validated on target ARM. Deeper pruning trades accuracy per the ft3 curve (20%: 78.11%, 30%: 68.04%); treat PELT 0.4 as the end of the steep regime, never an operating point. **Secondary:** a purpose-sized distilled student (25.8 ms, still inside the 50 ms budget) if the control loop tightens further.

### 9.4 Fari — companion device (< 20 MB, < 500 ms; consumer ARM SoC)

**Primary: no compression required — deploy the FP32 baseline as-is (93.38% / 6.24 MB / ~38 ms).** Both constraints are non-binding with wide margin, so accuracy decides — the uncompressed baseline wins outright. If headroom is desired, light pruning (10% + ft: 89.50% / 5.08 MB) buys ~19% size for 3.9 pp. In the secondary ranking under Fari-like constraints the pruned route also leads: at matched capacity (~1.7 MB), prune-50% (54.91%) beats the best distilled student (50.42%) by 4.49 pp — consistent with the inheritance mechanism of Section 7.4, though the training budgets differ, so indicative rather than settled.

### 9.5 Senpai — education robot, retrieval embeddings (< 20 MB; consumer ARM)

**Extrapolation, not a separate measurement.** Senpai's budget profile is closest to Fari's, so the Fari recommendation extrapolates: the FP32 baseline class fits as-is; light pruning is the headroom option. Two explicit caveats: (1) every number here comes from a *vision* CNN; embedding models are widely reported as more quantization-tolerant, so INT8 should be **re-tested, not presumed broken** for Senpai's workload (Section 10.3); (2) no Senpai-specific measurement exists in W1–W3. Treat this section as a testing plan, not a deployment sign-off.

### 9.6 Cross-platform summary

| Platform | Budget (MB / ms) | Primary recommendation | Ratio / bit-width | Expected profile (size / acc / lat) |
|---|---|---|---|---|
| Aido Rover | < 5 / < 100 | Light pruning + recovery ft | 12% (measured, W04 verification) | 4.87 MB / 88.45% / 46.6 ms (loaded session) |
| Sentinel Prime AI | < 3 / < 200 | Distillation route (production budget required) | width x0.5 student | 1.74 MB / TBD / 25.8 ms |
| Aido Humanoid | — / < 50 | Structured pruning <= 10%, with ft | <= 10% | 5.08 MB / 89.50% / 33.8 ms |
| Fari | < 20 / < 500 | Deploy FP32 as-is (constraints non-binding) | — | 6.24 MB / 93.38% / ~38 ms |
| Senpai | < 20 / — | Extrapolated from Fari; re-test INT8 on embeddings | — | Testing plan, not sign-off |

---

## 10. Limitations, Future Work, and Cross-Domain Notes

### 10.1 Measurement limitations

All latency was measured on an Apple M1 development CPU, not target hardware; absolute margins must not be trusted at 1:1, and the correct next step is **hardware-in-the-loop measurement** on Cortex-A silicon (and any RISC-V candidate — Section 10.5). Clean-session baseline latency spans 37.3–39.2 ms (+-3.0% about the mean; Section 3.4). The ft3 sweep's latency column was found load-contaminated from the 20% ratio onward and is superseded throughout this report by architecture-identical clean measurements (Section 3.4); the W04 verification-session latencies were taken under background load and are flagged wherever they appear. The scheduled re-measurement ran on 2026-07-30 (`scripts/remeasure_latency.py`): 12% = 43.79 ms, 15% = 42.20 ms, in-session baseline 49.61 ms — the control flags that session as loaded too, and the two loaded baselines agree (50.61 / 49.61 ms vs the 37.3–39.2 ms clean cluster), pointing to a persistent background condition. Both checkpoint measurements stand as session-annotated records; a verified-clean session was not obtained before sign-off, and no deployment conclusion depends on these absolute values. Accuracy and size reproduce exactly with seed 42.

### 10.2 Budget limitations

Recovery fine-tuning used a fixed 3 ep x 20k budget with training loss still falling at every ratio, so absolute accuracies likely understate longer-budget recovery (no held-out curve was tracked — expected, not verified); the changepoint location, rankings, and no-loss-free-region finding are statements under this budget, expected but not proven budget-robust. The distillation sweep used an exploration budget (5k images x 8 epochs); the +3.02 pp best-cell gain is tightly controlled (same data, budget, initialization) but is the maximum of nine single runs against one control run — the nine-cell mean sits 1.43 pp below control and no run-to-run variance was measured. Production deployment requires re-training at full-dataset, full-schedule budget. The Rover 12% recommendation rests on a Week-4 single-point measurement (Section 5.5) and shares the same budget caveat.

### 10.3 Scope limitations

Every number comes from one vision CNN (MobileNetV3-Small) on one dataset (CIFAR-10). Transfer to other families — Senpai's embeddings, Fari's dialogue models — is unvalidated; embedding/transformer models are widely reported as more quantization-tolerant, so the INT8 negative must not be generalized beyond this backbone. A mid-size transformer under the same eight-element protocol is the natural extension.

### 10.4 Toolchain and kernels

The quantization implementation uses `torch.ao` FX-mode, which PyTorch has marked deprecated in favor of torchao PT2E — production work should migrate. Absolute INT8 latencies are qnnpack numbers on Apple silicon; **hardware-specific kernels** (target-tuned INT8 GEMM/conv kernels) are required before any INT8 latency claim is transferred to deployment hardware. Ratios (size, relative speedup) transfer across hardware far better than absolutes.

### 10.5 A RISC-V-specific observation: why the recommendations are hardware-contingent

If Aido Rover's compute moves from an ARM Cortex-A core (NEON SIMD) to a RISC-V embedded core, **the recommendation that changes most is any quantization-based one.** The measured 11.3x INT8 speedup exceeds what either Section-2.3 mechanism predicts alone — ~4x from SIMD lane width (16 INT8 vs 4 FP32 lanes per 128-bit register), ~4x from bytes moved — so roughly 2.8x of it is unattributed: plausibly qnnpack's fused INT8 kernels against a less-optimized FP32 path, not isolated by profiling here. The ISA-contingent share is the vector-throughput part. On RISC-V that mechanism exists only with the V (vector) extension; on a base RV64GC core INT8 still buys the 4x size and traffic reduction but the throughput multiplier largely disappears — Sentinel's secondary recommendation would demote below pruning and distillation on a vector-less target, since INT8 then buys size but little speed. Pruning's size reduction is ISA-invariant and its latency gain survives any ISA, though absolutes must be re-measured. This is why batch-size-1 inference being **memory-bound** — the load/store cost model from RISC-V assembly coursework — is the most transferable insight here: on any ISA, moving fewer bytes is the reliable win; multiplying faster is the conditional one.

### 10.6 PELT across domains: the reservoir changepoints and the compression changepoint

The changepoint methodology of Section 5.4 transfers directly from the *Journal of Lake Sciences* study [7]. There, PELT segmented 2018–2021 monthly reservoir surface area (Wuhan-to-Hukou left-bank basin, Sentinel-1 SAR) into four regimes — slow decline, sharp decline from Sep 2018, a low plateau, then recovery — with the changepoints independently corroborated by changepoints in the SPEI12 drought index (Oct 2019; Jul 2020, +162.74%). Here, PELT segmented the accuracy-vs-ratio curve into two regimes at 0.4, supported by a penalty sweep and exact DP segmentation on the same signal.

Three correspondences carry across the domains. **(1) A changepoint marks a completed state transition, not the onset of damage:** reservoir area was already declining before Sep 2018, exactly as pruning already costs 3.9 pp at 10%; treating either changepoint as a "safe limit" misreads it. **(2) Changepoints deserve corroboration — and here the two projects differ honestly:** the reservoir study had genuinely external corroboration (SPEI12 — a different instrument, a different variable); this project has only internal robustness (penalty stability + Dynp on the same signal). Meeting that standard would need an independent signal changing regime at the same ratio — e.g. a per-layer feature-similarity metric. **(3) The criterion must be principled, not visual:** the study's penalty beta guards against over-segmentation; this project's criterion (penalty stability in [0.5, 2] + Dynp agreement) plays the same role. One honest difference: the study used an L2 segment cost, this project RBF — chosen because the curve's variance differs strongly between the steep-collapse and post-changepoint regimes. In both cases PELT marks qualitative structural change — a hydrological state shift there, the removal of entire feature-detector groups here.

---

## References

[1] inGen Dynamics — Products & PIC 2.0 (Origami) platform documentation, internship onboarding materials.
[2] Y. Cheng et al., "A Survey of Model Compression and Acceleration for Deep Neural Networks," arXiv:1710.09282.
[3] M. Nagel et al., "A White Paper on Neural Network Quantization," arXiv:2106.08295.
[4] H. Li et al., "Pruning Filters for Efficient ConvNets," arXiv:1608.08710.
[5] G. Hinton, O. Vinyals, J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531.
[6] A. Howard et al., "Searching for MobileNetV3," arXiv:1905.02244.
[7] Y. Liao, X. Ding, S. Xiang, et al., "Sentinel-1 observation on inundation dynamics of drinking water source reservoirs in the middle reaches of the Yangtze River," *Journal of Lake Sciences*, 2025.
[8] R. Killick, P. Fearnhead, I. A. Eckley, "Optimal Detection of Changepoints With a Linear Computational Cost," *JASA*, 2012 (PELT).

---

## Appendix A — Working Inventory (repository copy only; not part of the submitted report)

**Measurement decisions (final):**
1. **RESOLVED (2026-07-22): 12% pruning single point measured** — 88.45% / 4.87 MB (+ 15% bracket: 85.32% / 4.56 MB), folded into Sections 1, 5.3, 5.5, 6.1, 6.3, 9.1, 9.6, 10.2. Follow-up: commit `scripts/w04_rover_verification.py`, `experiments/W04_rover_verification.csv`, `rover_verify.log`, and the checkpoints (`rover_prune12_ft3.pt`, `rover_prune15_ft3.pt`) to the repository.
2. **REOPENED (2026-07-29; originally closed 2026-07-23): clean-session latency re-measurement.** The original close rested on "no deployment conclusion depends on an absolute latency." Two things changed since: the ft3 latency column was found load-contaminated (Section 3.4) — resolved by quoting architecture-identical clean sessions — and the 12%/15% checkpoints are now the flagship recommendation. The re-measurement via `scripts/remeasure_latency.py` ran 2026-07-30 and failed its in-session baseline control (49.61 ms vs the 37.3–39.2 ms clean cluster), so the 46.6 ms figure keeps its loaded-session flag and the re-measured 43.79 / 42.20 ms carry the same annotation (Section 10.1).

**Figure inventory:**
| # | Content | Source | Status |
|---|---|---|---|
| 1 | Baseline vs Rover deployment gap | `experiments/W01_deployment_gap.png` | EXISTS |
| 2 | Week-2 combined accuracy-vs-size Pareto | `experiments/W02_pareto_combined.png` | EXISTS |
| 3 | Per-layer activation ranges (stem outlier) | `experiments/W02_activation_ranges.png` (`scripts/w04_fig3_actranges.py`) | EXISTS |
| 4 | Pruning dual-axis ft3 + PELT | `experiments/W02_pruning_dualaxis_ft3.png` | EXISTS |
| 5 | Combined four-technique Pareto + budget lines | `experiments/W04_pareto_combined.png` (`scripts/w04_fig5_pareto.py`) | EXISTS |
| 6 | T x alpha heatmap | `experiments/W04_distill_heatmap.png` (`scripts/w04_fig6_heatmap.py`) | EXISTS |
| 7 | GA fitness + size per generation | `experiments/W03_nas_evolution.png` | EXISTS |

**Consistency item (resolved):** the repository README's front-page baseline now states both stages — the as-loaded ImageNet-head profile (10.31 MB / 40.46 ms) and the fine-tuned deployment baseline (93.38% / 6.24 MB) — matching Section 3.3 of this report.
