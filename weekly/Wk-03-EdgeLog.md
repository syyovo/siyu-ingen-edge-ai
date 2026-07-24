# Wk-03 EdgeLog — Knowledge Distillation, GA Micro-NAS, Compression Playbook

**What was compressed.** Knowledge distillation of the CIFAR-10
MobileNetV3-Small baseline (93.38% / 6.24 MB / 40.1 ms) into a
same-family width-0.5 student (409k params, −73%; 1.74 MB). Hinton loss
with a full T ∈ {2,4,8} × α ∈ {0.3,0.5,0.7} sweep (9 configs, 8 epochs ×
5k images each, seed 42), plus a from-scratch control at identical
architecture, data, budget, and init. Separately, a GA micro-NAS
(population 10, 15 generations, fitness = proxy accuracy − size penalty
under the < 5 MB Rover budget) ported from my orienteering-problem solver.

**What was measured.** Best distilled student: T=2/α=0.5 = 50.42% /
1.74 MB / 25.8 ms. From-scratch control: 47.40% — the distillation signal
is worth **+3.02 pp**, but only 2 of 9 configs beat the control (both at T=2); T=8/α=0.3
lands 8 pp below it. A first 3-epoch sweep underfit to 10.00% (archived);
a 60-step probe showed CE loss falling normally, so the failure was budget
starvation, not mechanics. GA-NAS best: 0.465 MB at 32.7% proxy accuracy,
with both fitness jumps on the same bottleneck gene — genuine
selection pressure, though it did not beat hand-designed baselines, as the
plan anticipated.

**Mechanism — best technique for Fari (< 20 MB, < 500 ms).** Fari's
budget is non-binding: every candidate fits, so accuracy decides. Among
the four compared candidates, pruning at the PELT ratio 0.4 wins (56.60%
vs 50.42%): a pruned model *inherits* the baseline's trained feature
detectors and loses only the removed channels' capacity, while a distilled
student must re-grow features from random initialization within a small
budget — the teacher's softened output distribution transfers inter-class
structure worth +3.02 pp, but cannot close the inheritance gap. Low
temperature (T=2) works best here because the teacher is already
confident; T=8 over-smooths its distribution into a weak training signal.
The honest Fari recommendation is therefore the FP32 baseline as-is
(93.38%), with light pruning as the headroom option — stated with full
evidence in the W03 playbook.