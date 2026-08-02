# Wk-04 Final EdgeLog — Verification, Reproducibility & Capstone

**What was measured.** No new compression technique this week — Week 4 verified
the existing ones. The 12% Rover recommendation, extrapolated in W2, was re-run
as a real experiment under the identical ft3 protocol
(`scripts/w04_rover_verification.py`): it lands at **88.45% / 4.87 MB** against
the ~88% prediction — verified within 0.5 pp — and ships inside the 5 MB budget
with 0.13 MB of headroom. The 15% run (**85.32% / 4.56 MB**) was a
bracketing point rather than a prediction under test: it prices deeper headroom
at ~1 pp per nominal percent across 12–15%.

**Mechanism.** The extrapolation held for a narrower reason than "the brackets
sit left of the PELT changepoint" — left of 0.4 guarantees nothing, since
degradation begins at the first ratio. It held because 12% sits two nominal
points from a measured anchor (10% = 89.50%) and the local slope is bounded:
0.53 pp per nominal percent over 10→12, 1.04 over 12→15, 1.44 over 15→20. The
curve is not near-linear — the local slope steepens with ratio — so a chord
through the 10% and 20% anchors under-predicts the interior, giving 87.2% and
83.8%, 1.2 and 1.5 pp below measurement. Short-range
interpolation from an adjacent measured point is reliable; interpolation across
the full 10–20% span is not.

**Reproducibility pass.** All five notebooks were re-run end-to-end
(Restart & Run All, seed 42), and every number the notebooks themselves compute
reproduced exactly — baseline 93.38% / 6.24 MB, INT8 per-channel 15.91%, PELT
changepoint 0.4 (56.60%; stable over penalty in [0.5, 2], none reported at
pen >= 3, Dynp agreeing on the same signal), KD best cell 50.42% vs 47.40%
control. Three headline numbers are re-read rather than re-computed at this
layer — the ft3 accuracies, QAT's 14.67%, NAS's 0.465 MB / 32.7% proxy —
because their source runs live in multi-hour scripts (`logs/`); the notebook
pass verifies the analysis layer, not those training runs.

**Data-audit correction.** Cross-checking the distillation table against its
summary caught a stale count: only **2 of 9** (T, alpha) configurations beat
the 47.40% control, not the "5 of 9" carried from an earlier control value.
Corrected in the report, deck, and Figure 6. The correction *strengthens* the
finding: the KD gain is hyperparameter-sensitive, never a default. A second
audit finding: the ft3 sweep's **latency column is load-contaminated** from the
20% ratio onward (32–75% above two architecture-identical clean sessions; the
sweep's own training wall-clock slows at the same ratios). All deployment-facing
latencies now quote clean sessions and the ft3 latency column is retired
(report Section 3.4). No conclusion changed; every latency margin widened.

**Status.** Capstone report, deck, and README finalized; repository tagged
v1.0; notebook layer fully reproducible. No blockers. Next: final presentation on
August 3 (30 min + 15 min Q&A).
