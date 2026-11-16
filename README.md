# Particle Transformer for Jet Tagging

> University coursework project for the **"Deep Learning for A.I."** course.
> Authors: **Manuel Bottino** and **Patrick Poetto** (group project, 2 co-authors).

This project implements, from scratch in PyTorch, a Transformer-based
classifier for **jet tagging** — one of the hardest and most actively
researched classification problems in experimental particle physics — and
trains it on **JetClass**, a large, modern public benchmark dataset for the
task. The architecture is directly inspired by **"Particle Transformer for
Jet Tagging"** (Qu, Li & Qian, 2022), re-derived and re-implemented by the
authors rather than copied from the paper's reference code.

## Contents

- [`jet_tagging_particle_transformer.py`](jet_tagging_particle_transformer.py) —
  the full project: data download/exploration, feature engineering, model
  architecture, training setup and the PyTorch Lightning training loop.
- This README: background, dataset, architecture, training setup and known
  results/limitations.

## What is jet tagging, and why does it matter?

At CERN's Large Hadron Collider (LHC), proton beams are collided at
energies close to the speed of light. These collisions can produce
short-lived, unstable particles (e.g. the Higgs boson, or the W, Z bosons
and top quarks) that decay almost instantly into a collimated spray of
more stable particles called a **jet**. Detectors cannot observe the
original particle directly — they only record the jet it leaves behind.

**Jet tagging** is the task of looking at a jet's reconstructed
constituent particles and inferring which (if any) high-interest particle
initiated it, versus the far more common "background" jets produced by
ordinary quark/gluon radiation (QCD). This is a foundational task for
essentially every physics analysis at the LHC: better jet tagging directly
translates into better sensitivity to new physics and more precise
measurements of the Standard Model.

<p align="center">
 <img src="https://i.imgur.com/ZYVGs2d.png" width="700">
 <br><i>High-energy proton-proton collision at the LHC. Source: arXiv:2202.03772</i>
</p>

## Why a Transformer?

A jet is naturally a **variable-size, unordered set (a "point cloud") of
particles** — typically 10 to 100 per jet — each carrying its own
kinematic and identity features. There is no inherent sequence order to
exploit, which makes jets a poor fit for RNNs/CNNs but a very natural fit
for **self-attention**: a Transformer encoder, applied with *no positional
encoding* (since the constituents are permutation-invariant), lets every
particle attend to every other particle in the jet and lets the network
learn which combinations of constituents are informative, rather than
hand-engineering physics features. This "Particle Transformer" (ParT)
approach, introduced by Qu, Li & Qian (2022), was state of the art on
JetClass at publication time and is the direct inspiration for this
project.

Paper: *["Particle Transformer for Jet Tagging"](https://arxiv.org/pdf/2202.03772.pdf)*,
Huilin Qu, Congqiao Li, Sitian Qian (arXiv:2202.03772).
Reference implementation referenced during development:
[`jet-universe/particle_transformer`](https://github.com/jet-universe/particle_transformer).

## Dataset: JetClass

[**JetClass**](https://zenodo.org/records/6619768) is a large, modern
public benchmark for jet tagging (introduced alongside the ParT paper). It
contains **10 jet classes**:

| Class | Decay | Description |
|---|---|---|
| `label_Hbb` | H → b b̄ | Higgs → bottom + anti-bottom |
| `label_Hcc` | H → c c̄ | Higgs → charm + anti-charm |
| `label_Hgg` | H → gg | Higgs → gluon + gluon |
| `label_H4q` | H → 4q | Higgs → 4 quarks |
| `label_Hqql` | H → l ν q q′ | Higgs → lepton + neutrino + 2 quarks |
| `label_Tbqq` | t → b q q′ | Top quark → bottom quark + 2 quarks |
| `label_Tbl` | t → b l ν | Top quark → bottom quark + lepton + neutrino |
| `label_Zqq` | Z → q q̄ | Z boson → quark + anti-quark |
| `label_Wqq` | W → q q′ | W boson → 2 quarks |
| `label_QCD` | — | Background (quark/gluon) noise |

For each particle in a jet, three groups of features are provided:
**kinematics** (4-momentum: energy, px, py, pz), **particle identification**
(charge + a 5-class one-hot: charged hadron / neutral hadron / electron /
muon / photon), and **trajectory displacement** (impact-parameter values
and errors, informative for tagging jets from bottom/charm quarks).

The full JetClass release provides 100M training jets, 5M validation jets
and 20M test jets, evenly split across the 10 classes. For this project:

- The dataset was explored using the official **100k-jet example file**
  (`JetClass_example_100k.root`) to understand the ROOT-file schema.
- The model was then **trained on ≈1,000,000 jets** sub-sampled from the
  full 100M-jet JetClass training partition (one file in every thirty,
  stratified so each of the 10 classes is represented), to keep the
  project tractable on Colab while remaining representative of the full
  dataset.

## Feature engineering

Raw per-particle features are transformed into the 17-dimensional input
vector used by the model, mirroring the feature set used in the ParT
paper's reference configuration: log(pt), log(energy), log(pt) and
log(energy) *relative to the jet*, ΔR to the jet axis, charge, the 5-class
particle-identity one-hot, tanh-compressed transverse/longitudinal impact
parameters (and their errors), and the raw Δη / Δφ to the jet axis. A
subset of these is standardized with fixed shift/scale constants taken
from the reference implementation. Every jet is zero-padded/truncated to a
fixed **128 constituent particles**, with a boolean mask marking the
padding.

## Model architecture

The implemented model (`ParT` in the code) follows the Particle
Transformer's overall design, re-implemented from the paper's description:

1. **Embedding** (`Embed`) — BatchNorm1d over the 17 input features,
   followed by a 3-layer MLP (`128 → 512 → 128`) with LayerNorm + GELU,
   projecting each particle into a 128-dimensional embedding. No
   positional encoding is added, since jet constituents are
   permutation-invariant.
2. **Particle attention blocks** (`P_MHA`, stacked `L=8` times via
   `L_blocks_P_MHA`) — pre-norm multi-head self-attention (8 heads) with a
   residual connection, followed by a pre-norm GELU MLP with another
   residual connection (a standard Transformer encoder block).
3. **Class attention blocks** (`ClassAttentionBlock`, stacked `L=2` times
   via `L_blocks_ClassAttentionBlock`) — a learnable **class token**
   (as in ViT/CaiT) attends over the particle tokens to pool them into a
   single, global, permutation-invariant jet-level representation.
4. **Classification head** (`Final`) — LayerNorm followed by an MLP
   (`128 → 100 → 50 → 10`) producing logits over the 10 jet classes
   (softmax applied only at inference time).

Hyperparameters were deliberately chosen to match the ParT paper's
published baseline configuration (`d=128` embedding, `L=8` particle-
attention blocks, `L=2` class-attention blocks, 8 attention heads,
dropout 0.1 on particle-attention blocks), confirmed by the training log
to total **≈353K trainable parameters**.

**Important, honestly-disclosed simplification:** the full ParT
architecture augments self-attention with a *pairwise particle-interaction
matrix* `U` (derived from relative kinematic/displacement features between
each pair of particles), added to the attention logits before the softmax:
`P-MHA(Q,K,V) = softmax(QKᵗ/√d + U)·V`. The authors of this project
explicitly attempted to reproduce this interaction block, found the
necessary implementation details underspecified in the paper, and
*omitted it* rather than guess — see the model-architecture section of the
code for their note on this. The model trained here is therefore the
attention-only "backbone" of ParT (particle self-attention + class-
attention pooling), without the interaction-aware attention bias. This is
a meaningful, disclosed architectural simplification relative to the full
published model, not an attempt to reproduce its exact published results.

## Training setup

- **Loss:** cross-entropy over the 10 one-hot jet classes.
- **Optimizer:** RAdam (lr `1e-3`, β=(0.95, 0.999), ε=`1e-5`) wrapped in
  **Lookahead** (α=0.5, k=6) — together equivalent to the "Ranger"
  optimizer used in the ParT paper.
- **Batch size:** 512 (train/val/test).
- **Epochs:** up to 15, with **early stopping** on validation loss
  (patience 2, min-delta `1e-2`) and **checkpointing** of the best model
  by validation loss.
- **Data split:** 64% train / 16% validation / 20% test (an 80/20 split,
  then an 80/20 split of the training portion), stratified implicitly by
  the class-balanced sampling used when building the ~1M-jet dataset.
- **Framework:** PyTorch Lightning, with a `TensorBoardLogger` used to
  track training/validation loss, accuracy and macro ROC-AUC live during
  training.
- **Metrics tracked:** accuracy, macro one-vs-one ROC-AUC, and — at test
  time — **background rejection at a fixed signal efficiency**
  (`Rej_X% = 1/FPR at TPR = X%`), the standard HEP metric for this task,
  computed per signal class against the QCD background class at the
  signal-efficiency thresholds used in the ParT paper.

## Results

The source material this project was reconstructed from (a text extraction
of the original Colab notebook) is **truncated partway through the first
training epoch** — specifically during PyTorch Lightning's initial
"sanity check" / early validation steps, right after the model summary
confirming **353K trainable parameters**. No final numeric results (test
accuracy, ROC-AUC, or per-class background-rejection scores) are present
in the recovered source, so **none are reported here** — in keeping with
not inventing results that weren't in the original material. The complete
training/evaluation pipeline (Lightning module, metrics, checkpointing,
TensorBoard logging, `trainer.test(...)`) is implemented and included in
the code; running it end-to-end on the full ~1M-jet dataset will produce
these metrics.

## Reconstruction / fidelity notes

This repository's code was reconstructed from a plain-text extraction of
the original `.ipynb` Colab notebook (not the raw notebook file itself).
The entire extraction was read and parsed; a few fidelity notes worth
flagging honestly:

- The extraction is cut off mid-way through a large, non-informational
  cached TensorBoard web-asset blob embedded by Colab in the final cell's
  metadata (the `%tensorboard` magic command) — i.e. a dump of JS/font
  binary data for offline rendering, not notebook content. Everything
  before that point — all 89 original cells — was fully recovered.
- Two code quirks were deliberately preserved rather than "fixed", since
  the brief was to preserve the authors' actual logic faithfully:
  the stacked particle-attention block (`L_blocks_P_MHA`) hardcodes
  `mask=None` internally (so the padding mask isn't applied there), and
  `ClassAttentionBlock.forward` contains a leftover debug `print(...)`.
- A couple of short architecture shape-probing cells near the end of the
  notebook were run out of order in the original interactive Colab
  session (referencing variables defined in a later cell) — preserved in
  their original order, commented, with a note in the code.

## Acknowledgements

- Qu, H., Li, C., & Qian, S. (2022). *Particle Transformer for Jet
  Tagging*. [arXiv:2202.03772](https://arxiv.org/pdf/2202.03772.pdf).
- JetClass dataset: [Zenodo record 6619768](https://zenodo.org/records/6619768).
- Reference implementation consulted: [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer).
