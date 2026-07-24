# W03 Compression Playbook — Per-Platform Recommendations
InGen Dynamics · Edge AI & Model Compression · Siyu Xiang · Week 3

**Scope.** Evidence-backed compression recommendations for the five InGen
platforms. All numbers come from the W01–W03 experiments on the
MobileNetV3-Small / CIFAR-10 baseline (93.38% / 6.24 MB / 40.1 ms — baseline 
row of experiments/W02_pruning_sweep_ft3.csv; the W02 quantization session
measured the same model at 38.5 ms; Apple M-series CPU, seed 42;
latency = bs=1, 10 warmup + 100 timed runs). Latencies from different sessions 
are load-sensitive (see session note in the W02 pruning notebook); accuracy 
and size reproduce exactly.

**Evidence base.** W02 PTQ INT8 (per-channel 15.91%, per-tensor 10.00%),
W02 QAT (14.67%), W02 9-ratio structured pruning sweep with 3-epoch
recovery, PELT changepoint at ratio 0.4 (56.60%), W03 distillation sweep
(9 configs, best T=2/α=0.5 = 50.42%), W03 from-scratch control (47.40%),
W03 GA micro-NAS (best found: 0.465 MB, proxy 32.7%).

**Cross-cutting finding.** A compact, efficiency-optimized architecture
resists post-hoc compression: INT8 collapses on an activation-range
pathology (hard-swish + SE) that neither fallback quantization nor a
small-budget QAT repairs; pruning has no loss-free region because there is
no dormant capacity. The stronger routes are light pruning with a recovery
budget, or training a smaller model directly (distillation — validated
direction, budget-limited in W3). Hardware qualifiers are stated per
recommendation because INT8 acceleration depends on the target ISA (SIMD
integer throughput, memory bandwidth).

---

## 1. Aido Rover — outdoor patrol, onboard perception
**(a) Deployment constraints.** < 5 MB, < 100 ms (10 Hz sensor stream:
each sample must be processed before the next arrives, with headroom).
Hardware class: ARM Cortex-A class embedded CPU.
**(b) Primary technique + mechanism.** Light structured pruning with
recovery fine-tuning: 10% + 3 ep ft = 89.50% / 5.08 MB / 34.2 ms —
0.08 MB over budget; ~12% pruning is extrapolated to clear 5 MB at ~88%
(flagged as extrapolation in the W02 memo, not a measurement). Mechanism:
pruning preserves trained feature detectors and removes only low-L1-norm
channels; latency scales with removed compute. Quantization is excluded
on this baseline: PTQ INT8 collapses to 15.91% (per-channel) and QAT does
not repair it (14.67%) — an activation-range pathology of hard-swish/SE,
not a calibration error. Note the INT8 *speed* mechanism itself is intact
(3.42 ms vs its same-session 38.5 ms baseline = 11.3×, per the W02
CSV — SIMD integer ALU throughput); the accuracy failure is
architecture-specific.
**(c) Recommended ratio + PELT evidence.** 10–12% pruning only. The PELT
changepoint sits at ratio 0.4 (56.60%) and marks *completed* collapse;
there is no loss-free region (10% already costs 3.9 pp), so every pruning
step must ship with a recovery budget.
**(d) Secondary technique.** Distilled width-0.5 student (1.74 MB /
25.8 ms) if the size budget ever tightens well below 5 MB — validated
route (+3.02 pp over same-budget from-scratch) but currently
budget-starved at 50.42%; requires a substantially larger training budget
before deployment.
**(e) Expected post-compression profile.** Prune 10% + ft: 5.08 MB /
89.50% / 34.2 ms (measured). Prune ~12% + ft: ~5.0 MB / ~88% (extrapolated).

## 2. Sentinel Prime AI — battery-powered patrol, always-on
**(a) Deployment constraints.** < 3 MB, < 200 ms, power-constrained
(5 Hz event classification). Hardware class: ARM edge CPU, battery budget.
**(b) Primary technique + mechanism.** Honest finding first: **no W2/W3
configuration deploys today.** Under 3 MB the measured options are
pruning @ 40% (2.37 MB but 56.60%) and INT8 (1.83 MB but collapsed).
The recommended *route* is knowledge distillation into a smaller student:
the width-0.5 student lands at 1.74 MB / 25.8 ms with headroom, and the
W3 controlled comparison shows the distillation signal is real (+3.02 pp
over from-scratch at identical architecture, data, budget, and init).
Mechanism: a student trained directly at the target capacity avoids the
post-hoc compression penalty entirely — but at the W3 budget (5k images ×
8 ep) it reaches only 50.42%: validated direction, insufficient budget
(same failure mode as the W2 pruning-recovery finding).
**(c) Recommended ratio + PELT evidence.** No pruning ratio satisfies
Sentinel: the budget-compliant region (≥ 40%, per the size curve) lies at
or past the PELT changepoint 0.4 — i.e. inside the collapsed regime.
**(d) Secondary technique.** Re-attempt quantization on a
quantization-friendly backbone (e.g. ReLU-based) rather than on
MobileNetV3; INT8's 3.42 ms / 1.83 MB profile is otherwise ideal for a
battery platform. This is a redesign recommendation, not a measured config.
**(e) Expected post-compression profile.** Distilled student at a
production training budget: 1.74 MB / 25.8 ms, accuracy TBD (must be
re-validated; 90%-class is the target, not a promise).

## 3. Aido Humanoid — bipedal research, joint-sensor sequences
**(a) Deployment constraints.** < 50 ms (10 Hz+ control loop). No
explicit size budget is defined in the W01 constraint table — this
platform is latency-critical, so latency is the binding constraint.
Hardware class: ARM with SIMD (NEON-class) — structured pruning chosen to
keep dense kernels SIMD-friendly.
**(b) Primary technique + mechanism.** Structured channel pruning is the
anchored technique, and the W2 result is a *negative* one to state
honestly: this baseline has no free redundancy — zero-shot pruning
collapses at every ratio, and even with 3-ep recovery, 10% costs 3.9 pp.
Mechanism: torch-pruning's dependency groups remove whole feature
detectors (channel + its BN + downstream couplings); on an
already-compact backbone every detector carries signal, so removal
creates an information bottleneck rather than trimming fat.
**(c) Recommended ratio + PELT evidence.** PELT changepoint = 0.4
(56.60%) under a stated criterion (RBF cost, penalty sweep,
Dynp-confirmed) — treat it as an upper bound, not an operating point.
Practical ceiling with the W2 recovery budget: ~10%.
**(d) Secondary technique.** Distillation into a purpose-sized student
(see Sentinel (b)); latency is the binding constraint here and the
student's 25.8 ms comfortably clears 50 ms.
**(e) Expected post-compression profile.** Prune 10% + ft: 5.08 MB /
89.50% / 34.2 ms — meets the 50 ms loop; deeper pruning trades accuracy
per the ft3 curve (20%: 78.11%, 30%: 68.04%).

## 4. Fari — low-latency companion
**(a) Deployment constraints.** < 20 MB, < 500 ms. Hardware class:
consumer-grade ARM SoC.
**(b) Primary technique + mechanism.** None required for this baseline:
the uncompressed FP32 model (93.38% / 6.24 MB / 40.1 ms) already fits
Fari's budget with wide margin — the constraints are non-binding, so
accuracy, not compression, is the deciding metric. If battery/memory
headroom is desired, light pruning (10% + ft: 89.50% / 5.08 MB / 34.2 ms)
buys ~19% size for 3.9 pp. Mechanism: as in §1(b).
**(c) Recommended ratio + PELT evidence.** ≤ 10% if any; PELT changepoint
0.4 marks completed collapse and no loss-free region exists.
**(d) Secondary technique.** Distillation (W3 four-way comparison):
best student 50.42% / 1.74 MB / 25.8 ms — relevant only if Fari's
footprint requirement ever tightens dramatically; needs a larger training
budget first. Among the four compared candidates under Fari's constraints,
the pruned model @ 0.4 wins on accuracy (56.60%): pruning inherits trained
features, while the student must re-grow them from random init within a
small budget — teacher soft labels recover +3.02 pp but cannot close the
inheritance gap. The KD gain is hyperparameter-sensitive: only 2 of 9
(T, α) configs beat from-scratch — both at T=2; T=8/α=0.3 lands 8 pp
below it (over-smoothed teacher signal). Caveat: Fari's production workload 
is language-model compression (small-BERT teacher, per the W01 table), while 
all evidence here comes from a vision CNN — the technique ranking is expected 
to transfer, the absolute numbers are not; re-validate on the language stack 
before deployment.
**(e) Expected post-compression profile.** Recommended (FP32 as-is):
6.24 MB / 93.38% / 40.1 ms. Alternative (10% pruned): 5.08 MB / 89.50% /
34.2 ms.

## 5. Senpai — education robot, course-retrieval embeddings
**(a) Deployment constraints.** < 20 MB (embedding-model quantization
context). No latency budget is defined in the W01 constraint table.
Hardware class: consumer ARM.
**(b) Primary technique + mechanism (extrapolated).** Senpai's budget
profile is closest to Fari's (< 20 MB), so the Fari recommendation
extrapolates: the FP32 baseline class fits the budget as-is; light
pruning (10% + recovery ft) is the headroom option. Mechanism: as in
§1(b). Extrapolation, not a separate measurement.
**(c) Recommended ratio + PELT evidence (extrapolated).** ≤ 10% if any —
the PELT changepoint 0.4 and the absence of a loss-free region were
measured on the shared vision baseline, not on a Senpai workload.
**(d) Secondary technique (testing plan).** Re-test INT8 on Senpai's
actual embedding model before ruling it out: every number in this
playbook comes from a *vision* CNN, and whether the hard-swish/SE
activation pathology transfers to embedding/transformer models is
unvalidated — embedding models are widely reported as more
quantization-tolerant.
**(e) Expected post-compression profile.** No Senpai-specific measurement
exists in W1–W3: treat this section as a testing plan, not a deployment
sign-off; baseline-class expectation if the extrapolation holds is the
~6 MB / ~40 ms class, accuracy workload-dependent.

---

## Cross-platform summary

| Platform | Budget (MB / ms) | Primary recommendation | Ratio / bit-width | Expected profile (size / acc / lat) |
|---|---|---|---|---|
| Aido Rover | < 5 / < 100 | Light pruning + recovery ft | 10–12% (12% extrapolated) | 5.08 MB / 89.50% / 34.2 ms — ~5.0 MB @ 12% (extrapolated) |
| Sentinel Prime AI | < 3 / < 200 | Distillation route (larger budget required) | width ×0.5 student | 1.74 MB / TBD / 25.8 ms |
| Aido Humanoid | — / < 50 | Structured pruning, ≤ 10%, with ft | ≤ 10% | 5.08 MB / 89.50% / 34.2 ms |
| Fari | < 20 / < 500 | Deploy FP32 as-is (constraints non-binding) | — | 6.24 MB / 93.38% / 40.1 ms |
| Senpai | < 20 / — | Extrapolated from Fari; re-test INT8 on embeddings | — | testing plan, not sign-off |

## Method note
Eight-element reporting per the Mars Trading sensitivity standard
(technique, ratio/bit-width, baseline acc, compressed acc, acc loss, size
before/after, latency before/after). PELT changepoint identified with
ruptures under a stated criterion, not by eye. All experiments seeded
(42) and reproducible from /experiments/ CSVs; failed runs (PTQ collapse,
QAT non-recovery, 3-ep distillation underfit, zero-shot pruning collapse)
are archived with mechanism notes rather than discarded.