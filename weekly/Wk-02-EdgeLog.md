# Wk-02 EdgeLog — Compression Experiments: Where the Cliff Actually Is

**What was tested.** Both post-hoc compression families against the
fine-tuned MobileNetV3-Small baseline (93.38% / 6.24 MB / 38.5 ms, seed 42,
CPU). Quantization: INT8 PTQ in four configurations plus 3-epoch QAT —
every one collapses to 10–16% accuracy despite real gains (1.83 MB, ~11×
faster). Pruning: structured L1 across nine ratios, zero-shot and with a
3ep×20k recovery budget. The week's real output is not a compressed model
but a pair of diagnosed failure mechanisms and a changepoint with a stated
criterion.

**The critical compression ratio, and what the changepoint actually
means.** PELT (RBF cost) on the recovery-budget curve places the
changepoint at nominal ratio 0.4 (56.60% accuracy), stable across penalty
∈ [0.5, 2] and independently confirmed by exact DP segmentation (Dynp).
But the textbook reading — "safe until the critical ratio, cliff after" —
is wrong for this model. The curve has no plateau: 10% pruning already
costs 3.9pt after a 12× recovery budget, and accuracy falls ~10pt per step
until 0.4, then ~4–5pt per step after. The changepoint marks where the
collapse *saturates*, not where it begins. Mechanism: dependency-group
coupling makes nominal ratios lie — nominal 40% physically removes 63% of
parameters (90% removes 98.6%) because residual/SE groups shrink together.
By 0.4, entire feature-detector groups are gone; the network hits its
residual capacity floor and there is little left to destroy. The
quantization collapse has the same root character — a distribution, not a
bug: weight-only quantization keeps 90.64%, but the stem activation spans
≈1986 against a network median of 14.95 (133×), so the INT8 grid step
(≈7.8) erases all normal-magnitude activations, and FP32 fallback cannot
help because the outlier tensor must still cross an INT8 boundary.
Both failures say the same thing: this architecture has no slack —
efficiency optimization already spent the redundancy that post-hoc
compression needs.

**Consequence for Week 3.** No Week 2 configuration reaches deployable
accuracy under the platform budgets: the closest, 10% pruning (89.5% /
5.08 MB), misses the Rover's 5 MB line by 0.08 MB, and nothing touches
Sentinel's 3 MB with usable accuracy — the Pareto chart's upper-left
region (>85% accuracy, <3 MB) is empty. Compressing a compact model
post-hoc failed twice for the same structural reason, so Week 3 inverts
the approach: instead of removing capacity from a trained network, train
a small network into competence — knowledge distillation of this 93.38%
baseline into a student sized for the Sentinel budget from the start.
Hardware qualifier, as always: latencies are Apple M / qnnpack numbers;
ratios transfer to Cortex-A, absolutes do not.
