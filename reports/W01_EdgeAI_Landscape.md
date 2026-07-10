# W01 — Edge AI Landscape Briefing

## 1. InGen Edge AI Context

InGen Dynamics builds physical AI platforms — robots that run AI models on
their own onboard hardware, not in the cloud. This creates a constraint that
software-only AI does not have: the model must fit and run on an embedded
processor with limited memory, limited compute, and often a battery.

The five platforms (Section 2) all face this constraint in different forms.
Aido Rover and Aido Humanoid run real-time control loops on Cortex-A-class
CPUs. Sentinel Prime AI is battery-powered, so energy per inference matters as
much as speed. Fari and Senpai run interactive companion / education systems
where response delay directly affects the user experience.

Model compression is the set of techniques that closes the gap between accurate
models (large, slow) and deployable models (small, fast). This briefing maps the
main techniques (Section 4), defines each platform's deployment budget
(Sections 2–3), and profiles an uncompressed baseline model to measure how far
it is from deployable (Section 5). It is the starting point for the compression
experiments in Weeks 2–3 and the per-platform playbook in Week 3.

## 2. Platform Constraint Table

| Platform | Deployment context | Max model size | Max latency | Primary compression strategy |
|---|---|---|---|---|
| Aido Rover (outdoor patrol) | 10 Hz real-time sensor classification on Cortex-A-class CPU | < 5 MB | < 100 ms | Quantization (W2 focus) |
| Sentinel Prime AI (security) | 5 Hz security-event classification, battery-powered | < 3 MB | < 200 ms | INT8 PTQ |
| Fari (elder companionship) | Compressed language model for dialogue | < 20 MB | < 500 ms | Knowledge distillation |
| Aido Humanoid (bipedal research) | Multi-step action classification, 10 Hz+ control loop | — (latency-critical) | < 50 ms | Structured pruning (SIMD on ARM) |
| Senpai (education robot) | Knowledge-base retrieval, embedding-model quantization | < 20 MB | — | Embedding quantization |

## 3. Hardware Rationale

### Why does a 10 Hz sensor stream require < 100 ms inference?

A 10 Hz sensor gives one new sample every 100 ms. So the model must finish one
inference before the next sample arrives. If inference takes longer than 100 ms,
samples come in faster than the model can process them. The unprocessed samples
keep piling up, and the robot ends up reacting to old data — for a patrol rover,
this means reacting to where an obstacle was, not where it is now.

Also, 100 ms is not only for the model. The same time window must also cover
sensor reading, preprocessing, and the decision logic after the model. So 100 ms
is a hard upper limit, and the real model budget should be clearly below it.

### Why does INT8 quantization improve latency on ARM edge CPUs, even without a GPU?

There are three hardware reasons. None of them needs a GPU.

**1. SIMD.** ARM CPUs have SIMD instructions (NEON on Cortex-A). One SIMD
instruction works on a 128-bit register. This register can hold 4 FP32 numbers,
but 16 INT8 numbers. So with INT8, one instruction does 4x more work.
Convolution is mostly multiply-accumulate (MAC) operations, and these map well
to SIMD instructions.

**2. Memory bandwidth.** An INT8 weight is 1 byte instead of 4 bytes. So the CPU
moves 4x less data from memory, and each cache line holds 4x more weights. This
matters a lot because edge inference at batch size 1 is usually memory-bound:
the CPU spends more time waiting for weights than computing. This is the same
idea from my RISC-V assembly course — memory access costs much more than
arithmetic.

**3. Integer ALU.** An integer multiply unit is a simpler circuit than a
floating-point unit (no exponent handling, no normalization). So integer
operations cost less energy and embedded cores usually run them faster. Lower
energy per inference also means longer battery life, which matters for
battery-powered platforms like Sentinel Prime AI.

One note: the speedup depends on the hardware. It is large on CPUs with wide
integer SIMD (ARM NEON, or RISC-V with the V extension), and much smaller on
CPUs without it. This is why every recommendation in this project states its
target hardware class.

## 4. Model Compression Taxonomy

### 4.1 Quantization
Quantization stores weights (and optionally activations) with fewer bits:
FP32 (4 bytes) down to INT8 (1 byte) or INT4. Model size drops almost 4x for
INT8, and inference gets faster on CPUs with integer SIMD (see Section 3).
Two main flavors: post-training quantization (PTQ) needs no retraining, only a
small calibration set to collect activation ranges; quantization-aware training
(QAT) simulates quantization during training and usually recovers more accuracy.
In this project: Week 2, PyTorch static PTQ on the baseline model, INT8 first,
INT4 if supported. Target platform: Aido Rover (< 5 MB budget — our 10.31 MB
baseline projects to ≈ 2.6 MB after INT8).

### 4.2 Structured Pruning
Pruning removes parts of the network. Structured pruning removes whole channels
or filters (not single weights), so the remaining model is a normal dense model —
it runs fast on standard hardware without special sparse kernels. Channels are
ranked by importance (for example L1 norm of the filter) and the weakest ones
are removed, usually followed by fine-tuning. Compression is adjustable: prune
10% for small savings, 90% for large savings with accuracy loss somewhere in
between — finding where accuracy collapses is the point of the Week 2 sweep.
In this project: Week 2, L1-norm channel pruning swept 10–90% in 9 steps, with
PELT changepoint detection on the accuracy curve. Target platform: Aido Humanoid.

### 4.3 Knowledge Distillation
Distillation trains a small "student" model to imitate a large "teacher" model.
The student learns not only the hard labels but the teacher's full output
distribution (softened by a temperature T), which carries information about how
classes relate to each other. The result is a small model that keeps more of the
teacher's accuracy than training the same small model from scratch. Unlike
quantization and pruning, distillation can change the architecture completely.
In this project: Week 3, Hinton (2015) loss with sweeps over T ∈ {2,4,8} and
α ∈ {0.3,0.5,0.7}. Target platform: Fari (language-model compression).

### 4.4 Neural Architecture Search (NAS)
NAS searches for a good architecture automatically instead of designing it by
hand. Hardware-aware NAS adds size or latency to the search objective, so the
search finds architectures that fit a deployment budget. Search can be done by
reinforcement learning, gradients, or evolutionary/genetic algorithms (GA).
MobileNetV3 — our baseline model — was itself found partly by hardware-aware NAS,
so Week 1 already benefits from this technique indirectly.
In this project: Week 3, a small GA-based micro-search over filter counts, with
model size constrained to the Aido Rover budget (< 5 MB).

## 5. Baseline Profiling Summary

Full details in `notebooks/W01_Baseline_Notebook.ipynb`. Selected model:
MobileNetV3-Small (ImageNet-pretrained, torchvision) + CIFAR-10 (via HuggingFace
datasets), matching the Aido Rover image-classification deployment context and
the Week-1 selection criteria (CPU-friendly size, public dataset, PyTorch
checkpoint).

| Metric | Baseline (FP32) | Aido Rover target | Status |
|---|---|---|---|
| Parameters | 2,542,856 | — | — |
| FLOPs (bs=1) | ≈ 113 M (56.5 M MACs) | — | — |
| Weight file size | 10.31 MB | < 5 MB | 2.1× over budget |
| CPU latency (bs=1, mean of 100) | 40.46 ms | < 100 ms | Passes on dev machine — see caveat |

**Deployment gap.** The main blocker is model size: 10.31 MB vs the 5 MB budget.
Latency passes on the development machine, but that machine is an Apple M-series
CPU, much faster than the Cortex-A-class CPU on Aido Rover, so the latency
margin cannot be trusted on real hardware.

**Week 2 plan.** INT8 quantization stores each weight in 1 byte instead of 4,
projecting 10.31 / 4 ≈ 2.6 MB — within budget. Structured pruning (10–90% sweep)
is the complementary technique.

## 6. Methodology Bridges

**Mars Trading sensitivity analysis → compression sweeps.** In my quant trading
internship, the core method was: vary one parameter over a range, measure two
competing metrics (win rate vs max drawdown), and map the Pareto trade-off. The
Week 2 compression experiments have exactly the same structure: vary the
quantization bit-width or pruning ratio, measure accuracy vs model size, and
find the Pareto-optimal configurations. The discipline is also the same — record
the whole trade-off curve, not just the best point.

**PELT changepoint detection → critical compression ratio.** In the Journal of
Lake Sciences study, we used the PELT algorithm to find where a reservoir's
surface-area time series changed structurally. In Week 2, the same algorithm
runs on a different signal: the accuracy vs pruning-ratio curve. The changepoint
it finds is the critical compression ratio — the point where accuracy stops
degrading gracefully and collapses, likely because entire feature detectors are
removed and create an information bottleneck.

## 7. References

**1. InGen Products & PIC 2.0 Documentation (onboarding materials)**
InGen Dynamics builds physical AI platforms (Origami / PIC 2.0) where models
must run inference on compute-limited embedded processors, not cloud servers.
The platform anchoring table defines two deployment constraints per platform:
max model size and max inference latency, driven by sensor rates and power
budgets — Aido Rover needs < 5 MB and < 100 ms on a Cortex-A-class CPU for
10 Hz sensor classification, while Sentinel Prime AI needs < 3 MB / < 200 ms
and Aido Humanoid needs < 50 ms for its 10 Hz+ control loop. These numbers are
the deployment targets for all compression experiments in this project; our
MobileNetV3-Small baseline (10.31 MB / 40.46 ms) already meets the Rover
latency budget but exceeds the size budget by about 2x, so size reduction is
the primary blocker.

**2. Cheng et al., A Survey of Model Compression and Acceleration for Deep
Neural Networks**
Problem: Deep networks are too large and too slow for devices with small
memory and strict latency limits. Method: The survey groups compression
methods into four categories: parameter pruning and quantization, low-rank
factorization, compact convolutional filters, and knowledge distillation.
Key result: A ResNet-50 example shows that removing redundant weights can cut
75% of parameters and 50% of compute time with no loss in function.
Relevance: This four-category taxonomy is the framework for the whole
project — it organizes the W1 landscape briefing and explains why pruning and
quantization (W2) work on pre-trained models while distillation (W3) needs
training from scratch. (arXiv:1710.09282)

**3. Nagel et al., A White Paper on Neural Network Quantization**
Problem: Neural networks trained in FP32 are too costly in memory traffic and
energy for edge devices. Method: The paper gives a hardware-grounded
introduction to uniform affine quantization (scale, zero-point, bit-width)
and provides tested standard pipelines for post-training quantization (PTQ)
and quantization-aware training (QAT). Key result: With their PTQ pipeline,
8-bit weight-and-activation quantization stays within 0.7% of FP32 accuracy
across all tested models, while INT8 cuts memory transfer by 4x and MAC cost
by about 16x. Relevance: This is the direct recipe for the W2 PTQ INT8
experiment — calibration set choice, symmetric weights with asymmetric
activations, and the warning that depthwise-separable models like our
MobileNetV3-Small can collapse under naive per-tensor PTQ without CLE or
per-channel quantization. (arXiv:2106.08295)

**4. Li et al., Pruning Filters for Efficient ConvNets**
Problem: Pruning individual weights creates irregular sparsity that needs
special sparse libraries, so it shrinks model files but rarely speeds up real
inference. Method: Rank filters in each layer by their L1 norm, remove whole
filters with the smallest norms together with their feature maps and
next-layer kernels, then retrain once to recover accuracy. Key result: On
CIFAR-10, this cuts VGG-16 FLOPs by 34% and parameters by 64% with no
accuracy loss, and the wall-clock speedup closely matches the FLOP reduction
because the pruned model stays dense. Relevance: This is the exact algorithm
for the W2 structured pruning sweep (L1-norm channel pruning at 10–90% plus
fine-tuning), and its per-layer sensitivity curves explain the mechanism
behind the PELT changepoint we will detect: accuracy collapses when pruning
starts removing filters from sensitive layers. (arXiv:1608.08710)

**5. Hinton et al., Distilling the Knowledge in a Neural Network**
Problem: Large models or ensembles are accurate but too heavy to deploy to
many users. Method: Train a small student model to match the teacher's
softened class probabilities, using a high softmax temperature T plus a
weighted hard-label loss. Key result: On MNIST, distillation cut a small
network's test errors from 146 to 74, close to the large teacher's 67.
Relevance: This is the exact loss used in the W3 distillation experiment
(L = alpha*CE + (1-alpha)*KL with T in {2,4,8}); the paper's finding that
intermediate temperatures work best for small students motivates our T sweep.
(arXiv:1503.02531)

**6. Howard et al., Searching for MobileNetV3**
Problem: Mobile vision needs models with high accuracy under tight CPU
latency and power limits. Method: Combine hardware-aware NAS with the
NetAdapt algorithm, plus manual improvements like the h-swish activation and
a redesigned last stage. Key result: MobileNetV3-Small reaches 67.4% ImageNet
top-1 with 2.5M parameters and 15.8 ms latency on a Pixel 1 CPU.
Relevance: This is our W1 baseline model itself; its quantization-friendly
design (h-swish avoids fixed-point precision loss) supports the W2 INT8 plan,
and our 10.31 MB FP32 checkpoint should shrink about 4x under INT8.
(arXiv:1905.02244)