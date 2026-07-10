# Wk-01 EdgeLog — Baseline Profiling & Hardware Constraints

**What was profiled.** MobileNetV3-Small (FP32, ImageNet-pretrained) as the
uncompressed baseline for the Aido Rover image-classification context:
2,542,856 parameters, ≈113 M FLOPs, 10.31 MB weight file, 40.46 ms CPU latency
(bs=1, mean of 100 runs, Apple M-series CPU). Deployment gap: size is the
blocker (10.31 MB vs < 5 MB, 2.1× over); latency passes on the dev machine, but
that margin cannot be trusted — the target is a much slower Cortex-A CPU.

**The hardware feature that most determines compression choice.** For
Aido Rover's CPU, it is memory bandwidth combined with integer SIMD width.
Edge inference at batch size 1 is memory-bound: each weight is loaded from DRAM,
used once, and not reused across a batch, so the CPU waits for data more than it
computes. This is the load/store cost model from my RISC-V assembly course —
memory access dominates runtime, not arithmetic. Quantization attacks exactly
this bottleneck: INT8 weights move 4× fewer bytes and fit 4× more values per
128-bit SIMD register (16 INT8 vs 4 FP32), so one NEON instruction does 4× more
work. A hardware qualifier applies: the speedup is large on cores with wide
integer SIMD (ARM NEON, RISC-V V extension) and much smaller without it.

**Consequence for Week 2.** Because the binding constraint is size and the
bottleneck is memory, INT8 PTQ is the first technique to test: projected
10.31 / 4 ≈ 2.6 MB, within the 5 MB budget. Structured pruning (10–90% sweep,
PELT changepoint) is the complementary experiment.