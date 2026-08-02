# W04 Retrospective — Edge AI & Model Compression Internship

**Siyu Xiang · inGen Dynamics · July 2026**

## The most surprising compression finding

I expected the negative results to be scattered failures. Instead they turned
out to be one finding with two faces: **a compact, efficiency-optimized
architecture resists post-hoc compression — each technique failing through its
own mechanism, both rooted in the same design choice.** INT8 PTQ collapsed to 15.91% not because quantization "is
lossy" but because one layer — the hard-swish stem — carries an activation
range 133× the network median, so 256 integer levels (255 steps) stretched over that span
erase every normal-magnitude activation. Pruning showed the mirror image: no
free region at all, with 10% already costing 3.9 pp, because MobileNetV3's
design (NAS-found widths, SE blocks) had already spent the redundancy that
pruning normally harvests. The model was pre-compressed by its designers; my
job turned out to be measuring exactly where the leftover slack ends.

## The most valuable technique from prior experience

**PELT changepoint detection.** Sensitivity sweeps (Mars Trading) produced the
data and the GA (orienteering solver) produced a working NAS sketch, but PELT
is what upgraded a plotted curve into a defensible claim. Instead of
eyeballing "accuracy drops around 40%," I could state a criterion: RBF-cost
PELT places the changepoint at ratio **0.4** (56.60%), stable across penalty
values in [0.5, 2], with exact dynamic-programming segmentation (Dynp)
agreeing on the same signal. That number then carried real weight downstream,
though not in the way I first wrote it up. The W4 brackets were chosen by the
Rover *size budget* — 10% pruning landed at 5.08 MB, 0.08 MB over, and ~12% was
the extrapolated ratio that cleared 5 MB — not because they sit left of the
changepoint. PELT ran the other way: it is why I never treated "left of 0.4" as
a safe region, and the measurements agreed — 12% costs 4.93 pp, 15% costs
8.06 pp.

## The PELT connection to the published research

In our *J. Lake Sci.* 2025 study (Liao, Ding, Xiang et al.), PELT was applied
to Sentinel-1 SAR time series of reservoir surface area to detect abrupt
structural change. The compression work applies the same algorithm to the same
class of signal, with a direct correspondence: the **time axis** maps to the
**pruning-ratio axis**; **surface area** maps to **test accuracy**; the
**drought-driven decline in reservoir area** maps to **capacity collapse** as
whole feature detectors are eliminated; and in both cases the changepoint was
established by a principled criterion rather than read off the plot —
corroborated from outside the curve by the SPEI12 drought index in the
reservoir study, and, more modestly, by internal robustness here (penalty
stability plus exact dynamic-programming agreement on the same signal; this
project has no truly external corroborating signal, an honest asymmetry
between the two studies).
The deepest shared lesson is interpretive: in both signals, PELT marks where
the transition has **completed** — where the new regime (the post-decline low
plateau there, the slower-but-still-falling decline here) is statistically
established — not where it begins. Reading the 0.4 changepoint as "safe up to
0.4" would repeat, in a network, the mistake of dating a drought by the month
the reservoir stopped falling.

## What I would do differently

Run the cheap diagnostic before the expensive treatment: the weight-only
quantization control (90.64%) took minutes and immediately located the failure
in activations — had it come first, I would have skipped one full PTQ variant
and gone straight to the range profile. Same lesson on the pruning side:
recovery fine-tuning is not optional polish but part of the technique, so its
budget belongs in the experiment design from day one, not discovered after a
failed 1-epoch probe.
