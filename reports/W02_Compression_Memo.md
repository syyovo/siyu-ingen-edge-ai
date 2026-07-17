# W02 Compression Memo — Quantization & Pruning of MobileNetV3-Small

**Author:** Siyu Xiang · Edge AI & Model Compression Intern
**Date:** July 17, 2026
**Scope:** INT8 quantization (PTQ + QAT) and structured L1 pruning of the CIFAR-10 baseline; critical-ratio analysis (PELT); deployment recommendations per platform.

---

## 1. Executive Summary

Week 2 benchmarked the two post-hoc compression families on the fine-tuned MobileNetV3-Small baseline (93.38% / 6.24 MB / 38.5 ms). **Both fail on this architecture, and both failures are diagnosed, not merely observed.** INT8 quantization delivers real size (3.4×) and speed (~11×) gains but collapses accuracy under every PTQ configuration and under QAT; a controlled experiment shows weight quantization alone is nearly lossless (90.64%), isolating the cause to an activation-range pathology — the stem output spans ≈1986 against a network median of 14.95 (133× outlier) — that selective FP32 fallback cannot bypass because the outlier tensor must still cross an INT8 boundary. Structured pruning shows no free region either: even 10% pruning costs 3.9pt after a 3ep×20k recovery budget, and PELT places the changepoint at ratio **0.4** (56.60%), marking where collapse saturates rather than where it begins. No Week 2 configuration reaches "deployable accuracy at < 5 MB"; the closest is 10% pruning (89.5% / 5.08 MB), 0.08 MB over the Aido Rover budget. **Conclusion: this compact, efficiency-optimized architecture resists post-hoc compression. For the Sentinel Prime budget (< 3 MB), the viable path is a smaller architecture trained directly — knowledge distillation, Week 3's topic.**

| Config | Test acc | Size (MB) | Latency (ms, bs=1) |
|---|---|---|---|
| FP32 baseline (fine-tuned) | 93.38% | 6.24 | 38.5 |
| INT8 PTQ per-channel | 15.91% | 1.83 | 3.42 |
| INT8 PTQ per-tensor | 10.00% | 1.65 | 3.36 |
| INT8 QAT (3 ep) | 14.67% | 1.83* | 3.42* |
| Prune 10% + 3ep×20k ft | 89.50% | 5.08 | 34.2 |
| Prune 50% + 3ep×20k ft | 54.91% | 1.72 | 33.7† |
| Best deployable config | **none meets budget + accuracy; closest: prune 10%** | 5.08 | 34.2 |

*\* QAT produces the same quantization scheme as per-channel PTQ; size/latency measured on the per-channel artifact and reused.*

*Protocol: seed 42, CPU (Apple M, PyTorch 2.12.1), latency = bs=1, 10 warmup + 100 timed runs. Calibration: 512 train-split images. All accuracy on the full 10k test split unless marked. Latencies are quoted from their respective measurement sessions; session-to-session variation on the shared dev machine is ±5–10% under clean conditions — except the final pruning-notebook rerun, which ran under background load (flagged in that notebook).*

*† Noisy measurement (std 8.18 ms; taken under background load in the ft3 session). The same architecture measured 19.8–25.8 ms in a later session that was itself partially loaded, so the true idle-machine figure is likely in the low-20s; flagged for re-measurement rather than over-claimed.*

## 2. Baseline

- Fine-tuned MobileNetV3-Small (ImageNet weights → 10-class head, 2 ep head-only + 2 ep full, Adam): **93.38% / 6.24 MB / 38.5 ms / 1,528,106 params**.
- Note vs W01: swapping the 1000-class ImageNet head for a 10-class head shrank the model from 10.31 MB to 6.24 MB (~1M params were in the old head alone). The true deployment gap to the Aido Rover budget (<5 MB) is therefore 1.24 MB, not the 2.1x reported in W01.

## 3. Quantization Findings

### 3.1 PTQ collapses under every configuration

Static PTQ INT8 (FX graph mode, qnnpack) delivers the expected size (3.4x → 1.83 MB) and speed (~11x) gains, but accuracy collapses to near-random under all four tested configurations:

| Weight scheme | Activation observer | Test acc |
|---|---|---|
| per-channel | Histogram | 15.91% (full test set) |
| per-tensor | Histogram | 10.00% (full test set) |
| per-channel | MinMax | 12.2% (1k quick set) |
| official default mapping (qnnpack) | — | 10.2% (1k quick set) |

Direction matches Nagel et al. (arXiv:2106.08295): per-tensor < per-channel on depthwise-separable networks. Magnitude exceeds it: even per-channel collapses.

### 3.2 Root-cause diagnostic chain

1. **Prediction distribution:** INT8 model dumps ~78% of a test batch into 3 of 10 classes → systematic output distortion, not mere noise.
2. **Single-conv sanity check:** one Conv2d+ReLU quantized in isolation shows max abs error ≈ 0.30 on outputs in [0,4] → kernels compute correctly; toolchain bug ruled out.
3. **Observer swap (Histogram → MinMax):** still collapsed → calibration-strategy hypothesis ruled out.
4. **Official default qconfig mapping:** still collapsed → misconfiguration hypothesis ruled out.
5. **QAT cross-check:** training loss converges (1.90 → 0.58) but simulated-INT8 accuracy (11.8%) matches true-INT8 accuracy (11.6%) → the failure exists in simulation, independent of the qnnpack backend.
6. **Experiment A (weight-only quantization): 90.64%** on the full test set (−2.74pt vs baseline) → weight quantization is essentially lossless; **the collapse is caused by activation quantization.**
7. **Experiment B (per-layer activation ranges):** the stem convolution output (`features.0.0`) spans **−1073.9 … 912.2 (range ≈ 1986)** against a network-wide median of **14.95 — a 133× outlier**. Its INT8 step is ≈ 7.8, so all normal-magnitude activations (|x| ≲ 4) collapse into a handful of bins; the wrecked stem output propagates network-wide, explaining the class-collapse in step 1. Secondary outliers (range 65–118, 4–8× median) span features.1–.7 and features.12.

**Fix attempt — selective FP32 fallback: 14.7% (1k quick-screen; failed, verified).** Keeping the stem block (`features.0`, covering both top outliers) in FP32 does not recover accuracy, because the pathology lives in *tensors*, not ops: the stem's output must still be quantized at the boundary into the next INT8 block, and that boundary grid is set by the ≈912-range outlier. Escaping it would require extending the FP32 region through features.1–.7, forfeiting most of the compression benefit. Module-level inspection confirms the fallback itself took effect — this is a genuine negative result, closed under a pre-committed time-box.

**Likely origin of the outliers:** after only 4 epochs of fine-tuning, the stem retains near-ImageNet weights while the input distribution changed to CIFAR-10 upscaled 32→224; some channels respond with extreme values, and min/max calibration faithfully stretches the grid over them — the activation-outlier failure mode described for efficient architectures by Nagel et al.

### 3.3 QAT

QAT (per-channel FakeQuantize, lr 1e-4, 3 epochs from the 93.38% baseline; implementation: `scripts/qat_train.py`, metrics in `checkpoints/qat_summary.json`) converged in training but did not recover INT8 accuracy (best 14.67%), consistent with the simulation-level failure in 3.2: a small-budget weight adaptation cannot fix an activation-distribution pathology of this magnitude. Reported as a confirmatory negative result.

**INT4** was not attempted: no CPU INT4 inference path exists in this toolchain on macOS (bitsandbytes is CUDA-oriented). Mechanistically, a 2× coarser grid can only worsen the same activation-range pathology, so the negative conclusion extends a fortiori.

## 4. Pruning Findings

### 4.1 Zero-shot: no free redundancy

Structured L1 pruning (torch-pruning, dependency-graph aware; classifier head excluded). Zero-shot accuracy collapses at even 10% channel removal (93.4% → 15.0%), and the whole 10–90% curve sits at random (~10%). Two mechanisms:

- MobileNetV3-Small is already efficiency-optimized; unlike over-parameterized networks, it carries no dormant channels.
- Dependency-group coupling amplifies nominal ratios: nominal 10% removes **18.8%** of parameters, nominal 40% removes **63%**, nominal 90% removes **98.6%** (1,528,106 → 21,948 params; residual/SE groups shrink together).

Size/latency gains are real: 50% → 1.72 MB, 90% → 0.17 MB; latency falls with parameters but sub-linearly (90% pruning: 6.8–7.1 ms across sessions vs 38.5 ms baseline), consistent with W01's memory-bound finding for bs=1 inference. Absolute per-ratio latencies varied across measurement sessions (see §1 protocol note) and are quoted as indicative rather than precise.

### 4.2 Recovery-budget experiment

1 epoch × 5k images recovers 10%-pruning to only 55.0% and 20%+ not at all — that curve measures the recovery budget, not the architecture. The 12× budget rerun (3 epochs × 20k images, all nine ratios) confirms recovery is real at low ratios but there is **no loss-free region**: even 10% costs 3.9pt.

| Ratio | Zero-shot acc | +1ep/5k acc | +3ep/20k acc (ft3) |
|---|---|---|---|
| 10% | 15.04% | 55.01% | **89.50%** |
| 20% | 11.23% | 12.83% | **78.11%** |
| 30% | 7.63% | 12.38% | **68.04%** |
| 40% | 9.96% | 9.71% | **56.60%** |
| 50% | 10.54% | 11.27% | **54.91%** |
| 70% | 10.00% | 10.00% | **46.54%** |
| 90% | 10.00% | 10.00% | **33.71%** |

*Full nine-ratio data in `experiments/W02_pruning_sweep_ft3.csv` (60%: 44.88%, 80%: 40.86%). The 60→70% uptick (44.88 → 46.54) is fine-tuning noise at small parameter counts. Zero-shot and +1ep columns are from the final rerun of the pruning notebook; the ft3 column is from `scripts/prune_ft_sweep.py`. Training loss was still falling at epoch 3 in every ratio, so this curve is a lower bound under the stated budget; the shape and conclusions are budget-robust, absolute values are not.*

### 4.3 Critical compression ratio (PELT)

**Changepoint: nominal ratio 0.4, test accuracy 56.60% at the changepoint.**
Criterion: PELT (ruptures, RBF cost) is stable at 0.4 across penalty ∈ [0.5, 2] and reports no alternative location at any penalty; cross-validated by exact dynamic-programming segmentation (Dynp, n_bkps = 1), which independently selects 0.4.

**Mechanism.** The changepoint marks the transition from steep collapse to saturation, not the onset of damage: before 0.4 accuracy falls ~10pt per step (93.4 → 56.6); after it, ~4–5pt per step. By nominal 0.4, dependency-group coupling has physically removed 63% of parameters — entire feature-detector groups are gone, creating an information bottleneck that 3-epoch fine-tuning cannot rebuild, and accuracy approaches the residual capacity floor. Critically, **the region before the changepoint is not a safe zone**: degradation begins at the first step (10% already costs 3.9pt). Unlike over-parameterized networks, this curve has no pre-changepoint plateau — 0.4 is not a "safe limit" but the point where there is little left to destroy.

Dual-axis figure: [`../experiments/W02_pruning_dualaxis_ft3.png`](../experiments/W02_pruning_dualaxis_ft3.png) (accuracy + size vs ratio, changepoint marked).

## 5. Pareto Frontier & Platform Recommendations

Combined accuracy-vs-size scatter of all Week 2 configurations, colored by latency: [`../experiments/W02_pareto_combined.png`](../experiments/W02_pareto_combined.png). The frontier runs from the FP32 baseline down the pruning-recovery chain, with the INT8 points occupying the small-size / collapsed-accuracy corner.

| Platform | Budget | Recommended config | Meets budget? |
|---|---|---|---|
| Aido Rover (Cortex-A + RISC-V) | <5 MB, <100 ms | Prune 10% + recovery ft: 89.50% / 5.08 MB / 34.2 ms | **Almost** — 0.08 MB over. Extrapolated ~12% pruning clears the size budget at an estimated ~88% accuracy (not measured; flagged as extrapolation). |
| Sentinel Prime AI (edge, always-on) | <3 MB, <200 ms | **No W2 config is viable.** First pruning point under 3 MB (40%: 2.37 MB) has 56.60% accuracy; INT8 meets size (1.83 MB) with collapsed accuracy. | **No** — distillation (W3) is the only credible route. |

Hardware qualifiers to preserve in every claim: absolute latencies are Apple M / qnnpack numbers; ratios (size, speedup) transfer to Cortex-A better than absolutes; INT8 speedup depends on integer-SIMD width of the target.

## 6. Honest Limitations & Week 3 Implications

- All latency measured on a development machine (Apple M), not target hardware; margins must not be trusted at 1:1. The prune-50% figure is noisy (33.7 ms, std 8.18, background load in the ft3 session; a later, itself partially loaded session measured 19.8–25.8 ms) and is flagged for re-measurement rather than over-claimed.
- Recovery fine-tuning used a fixed 3ep×20k budget and had not fully converged at any ratio; the ft3 curve is a lower bound under that budget. The changepoint location and the "no free region" finding are expected to be budget-robust; absolute recovered accuracies are not.
- The 10%-pruning result (89.50% / 5.08 MB) suggests light pruning + larger recovery budget could close the Rover gap; the ~12% recommendation is an extrapolation, not a measurement.
- The central Week 2 lesson: **a compact, efficiency-optimized architecture resists post-hoc compression** — PTQ collapses on an activation-range pathology that fallback cannot bypass, QAT cannot repair it within a small budget, and pruning has no loss-free region because there is no dormant capacity to remove. For platforms needing <3 MB, the stronger path is a smaller architecture trained directly, or knowledge distillation into a smaller student — Week 3's topic.
- Deprecation note: implementation uses torch.ao FX-mode quantization, marked deprecated in favor of torchao PT2E; production work should migrate.
