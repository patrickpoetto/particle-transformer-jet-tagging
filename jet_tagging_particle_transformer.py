"""
Particle Transformer for Jet Tagging
=====================================

University coursework project for the "Deep Learning for A.I." course.
Authors: Manuel Bottino, Patrick Poetto.

Dataset : JetClass (Qu, Li & Qian, 2022) - https://zenodo.org/records/6619768
Paper   : "Particle Transformer for Jet Tagging" - https://arxiv.org/pdf/2202.03772.pdf
Reference implementation used for comparison / data utilities:
          https://github.com/jet-universe/particle_transformer

See README.md for the full write-up (physics background, dataset description,
architecture summary, training setup and known limitations).

--------------------------------------------------------------------------
RECONSTRUCTION NOTE
--------------------------------------------------------------------------
This script was reconstructed from a text extraction of the original Google
Colab notebook ("4 - Deep Learning for A.I. - Particle Transformer Neural
Network for Jet Tagging.ipynb"). The extraction was read in full; it turned
out to be truncated mid-way through a (large, non-informational) cached
TensorBoard web-asset blob that Colab had embedded in the metadata of the
very last cell (the `%tensorboard` magic command). Everything before that
point -- all 89 original cells (37 markdown, 52 code): environment setup,
data download/exploration, feature engineering, model definition,
hyperparameters, Lightning training module, data loaders and the trainer
launch -- was fully recovered and is faithfully reproduced below, including
the authors' original comments (even a couple of rhetorical "why/what"
notes they left themselves while reverse-engineering the paper).

Because the extraction cuts off while the first training epoch is still in
its "Sanity Checking" / early validation steps, no final numeric results
(test accuracy, ROC-AUC, background-rejection scores) are present in the
source. They are therefore NOT reported here or in the README -- see the
"Results" section of the README for what is and isn't known.

Two small fidelity notes on the code itself (kept as originally written,
not "fixed", per the reconstruction brief):
  * In `L_blocks_P_MHA.forward`, the `mask` argument is accepted but each
    internal block call hardcodes `mask=None`, so the attention mask is
    effectively not applied inside the stacked particle-attention blocks.
  * `ClassAttentionBlock.forward` contains a leftover debug `print(...)` of
    intermediate tensor shapes.
  * The short "sanity check" cells near the end of the notebook (shape
    probes of Embed/MHA/Linear building blocks) were run out of order in
    the original interactive Colab session: an early demo cell reuses
    variables (`out0`, `cls`, ...) that are only defined in a later cell.
    The original relative order is preserved below with a note, rather
    than silently reordered.
"""

# ==========================================================================
# 0. Environment setup (Colab-specific)
# ==========================================================================
# Original notebook mounted Google Drive, since the JetClass data and model
# checkpoints were stored there across sessions.

from google.colab import drive
drive.mount('/content/drive')

# ==========================================================================
# 1. Dependencies
# ==========================================================================
# %%capture
# Run if you have not yet run it once
# !pip install awkward uproot vector pytorch_lightning xxhash lz4

import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()

import os
import shutil
import zipfile
import tarfile
import urllib
import requests
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

import collections
import hashlib
import math
import random
import re
import sys
import time
from collections import defaultdict
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


# ==========================================================================
# 2. Download & explore the JetClass example dataset
# ==========================================================================
def _download(url, fname, chunk_size=1024):
    """https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51

    Downloads a file from `url` and saves it locally as `fname`, streaming
    it in chunks (`chunk_size` bytes) and showing a tqdm progress bar sized
    from the response's `content-length` header.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


# Download the small, official 100k-jet example file (used here purely for
# exploring the schema of the dataset) if it isn't already present.
example_file = 'JetClass_example_100k.root'
os.chdir("/content/drive/MyDrive/JetClass/data")
if not os.path.exists(example_file):
    _download(
        'https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root',
        example_file,
    )

# --- Exploring the file --------------------------------------------------
# Load the content from the file
tree = uproot.open(example_file)['tree']

# Display the content of the "tree" (branch names / types)
tree.show()

# tree.arrays() retrieves all the data from the tree in the ROOT file and
# organizes it into a Python dictionary of NumPy / Awkward arrays. Each key
# corresponds to a branch name in the ROOT file.
table = tree.arrays()

# Arrays of a scalar type (bool/int/float) can be converted to a numpy array directly, e.g.
# table['label_QCD'].to_numpy()

# Arrays of a vector type are loaded as a JaggedArray with varying elements
# per row (number of constituent particles varies jet to jet). A JaggedArray
# can be (zero-)padded to become a regular numpy array (see `_pad` below).
# table['part_px']

# Construct a Lorentz 4-vector from the (px, py, pz, energy) arrays. Each
# vector represents the four-momentum of a particle (momentum along x, y, z
# and energy), the standard way of characterising particles in HEP analyses.
p4 = vector.zip({
    'px': table['part_px'],
    'py': table['part_py'],
    'pz': table['part_pz'],
    'energy': table['part_energy'],
})

# p4.pt / p4.eta / p4.phi derive the transverse momentum, pseudorapidity and
# azimuthal angle of each particle from the 4-vector.
# p4.pt, p4.eta, p4.phi


# ==========================================================================
# 3. Feature engineering: padding, clipping, and building model inputs
# ==========================================================================
def _pad(a, maxlen, value=0, dtype='float32'):
    """Pads (or truncates) arrays or arrays-within-arrays to a fixed length `maxlen`.

    - If `a` is already a 2D numpy array whose second dimension equals
      `maxlen`, it is returned unchanged.
    - If `a` is an Awkward Array, it is (optionally first unflattened to
      2D, then) padded with `ak.pad_none` / `ak.fill_none` to `maxlen`
      and cast to `dtype`.
    - Otherwise (e.g. a plain list of variable-length sequences), a dense
      numpy array filled with `value` is allocated, and each row is
      truncated to `maxlen` or left zero/`value`-padded.

    Used to turn the variable-length, per-jet list of constituent particles
    into a fixed-size (maxlen x n_features) tensor the model can consume.
    """
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, :len(trunc)] = trunc
        return x


# Apply zero-padding and convert to a numpy array, e.g.:
# _pad(p4.pt, maxlen=128).to_numpy()


def _clip(a, a_min, a_max):
    """Clips array values to the [a_min, a_max] range.

    Tries plain `np.clip` first; if that raises (because `a` is a jagged
    Awkward Array of variable-length rows, which numpy can't clip directly),
    falls back to flattening with `ak.flatten`, clipping, and restoring the
    original jagged structure with `ak.unflatten`.
    """
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def build_features_and_labels(a, transform_features=True):
    """Builds the (points, features, vectors, mask) input tensors and the
    one-hot label tensor expected by the Particle Transformer, following the
    feature construction described in the ParT paper / reference repo
    (see https://github.com/jet-universe/particle_transformer/blob/main/data/JetClass/JetClass_full.yaml).

    Main steps:
      - Load the needed arrays from the input tree/table `a`.
      - Compute derived kinematic features (log pt, log energy, log-relative
        pt/energy, delta-R to the jet axis, tanh-compressed impact
        parameters, ...).
      - Standardize a subset of those features (if `transform_features`)
        using fixed shift/scale constants taken from the reference
        implementation.
      - Pad every per-particle feature to a fixed `maxlen=128` constituents
        per jet and stack them into `pf_points`, `pf_features`, `pf_vectors`
        and `pf_mask` arrays.
      - Stack the 10 one-hot class labels into a single `label` array.
    """
    # load arrays from the tree, where * stand for all of the filters that contains
    # what is before the *

    # remove comment when 100k example
    # a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt'] / a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy'] / a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['part_d0'] = np.tanh(a['part_d0val'])
    a['part_dz'] = np.tanh(a['part_dzval'])

    # apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0
        a['part_d0err'] = _clip(a['part_d0err'], 0, 1)
        a['part_dzerr'] = _clip(a['part_dzerr'], 0, 1)

    # Define the skeleton of the output dict
    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'],  # not used in ParT
        'pf_features': [
            # Kinematics: part_pt includes the momentum except pz (WHY PZ NOT COMPUTED ABOVE??)
            # the rest are kinematics that can be computed given the pf_vectors
            # measured in units of GeV (WHAT???)
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',

            # 5 class onehot encoding of the particle identification (value +-1).
            # Also, it takes into consideration the particle identity determined by
            # the detector system
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',

            # values and errors of the transverse and the longitudinal impact
            # parameters of the particle trajectories in units of mm
            # These values are only present for charged particles, the neutral ones
            # are assigned 0
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',

            # Explanation: Pseudorapidity and azimuthal angle
            # difference between the particle and jet.
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask'],
    }

    # Create the dictionary that will be the output of the function
    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    label_list = ['label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql',
                  'label_Tbqq', 'label_Tbl', 'label_Zqq', 'label_Wqq', 'label_QCD']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out


# ==========================================================================
# 4. Building the full training dataset (~1M jets)
# ==========================================================================
# We've imported the Git repository containing the dataset, which comprises
# batches with 10 million observations. Each batch consists of 10 files per
# class (10 classes), with each file containing 100,000 observations. Due to
# various challenges, we opted to work with 1 million observations, ensuring
# data integrity and avoiding potential issues. To maintain an unbiased
# model, we carefully selected one batch, encompassing 100,000 observations,
# from each distinct class.

# !git clone https://github.com/jet-universe/particle_transformer.git

# os.chdir("/content/particle_transformer")
# !/content/particle_transformer/get_datasets.py JetClass -d "/content/drive/MyDrive/JetClass/data"

import glob

file_pattern = "/content/drive/MyDrive/JetClass/data/JetClass/Pythia/train_100M/*.root"

# glob to find all files matching the pattern in a specific directory
file_list = sorted(glob.glob(file_pattern))  # sort to ensure it maintains the order in the folder
file_list = file_list[::30]  # 1M, add + file_list[1::30] for 2 Million

# from tree to array, then concatenate all of them with ak
trees = ak.concatenate([
    uproot.open(file)["tree"].arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])
    for file in file_list
])

data = build_features_and_labels(trees)

# use pickle together with gzip to reduce the dimension of big file and write in the drive
import pickle
import gzip

file_path = '/content/drive/MyDrive/JetClass/data/JetC_1M.pkl.gz'
# Compress and save the dictionary
with gzip.open(file_path, 'wb') as file:
    pickle.dump(data, file)


# ==========================================================================
# 5. Load dataset and split into train / validation / test
# ==========================================================================
# use pickle together with gzip to read big files stored previously in the drive
file_path = '/content/drive/MyDrive/JetClass/data/JetC_1M.pkl.gz'

# Retrieve the compressed dictionary
with gzip.open(file_path, 'rb') as file:
    data = pickle.load(file)

df = data.copy()

# from the imported dictionary store all the important variables in different variables
labels = torch.from_numpy(df["label"])
features = torch.from_numpy(df["pf_features"])
mask = torch.from_numpy(df["pf_mask"])
momentums_energy = torch.from_numpy(df["pf_vectors"])
pf_points = torch.from_numpy(df["pf_points"])

labels = torch.tensor(labels, dtype=torch.long)

# Create train, test and val. We achieve it by first splitting train and testing
# then we split the train to obtain train and validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, train_mask, test_mask = train_test_split(
    features, labels, mask, test_size=0.2, train_size=0.8, random_state=42)
X_train, X_val, y_train, y_val, train_mask, val_mask = train_test_split(
    X_train, y_train, train_mask, test_size=0.2, train_size=0.8, random_state=42)

assert train_mask.shape[0] == X_train.shape[0] == y_train.shape[0]


# ==========================================================================
# 6. Model architecture: Particle Transformer (ParT)
# ==========================================================================
# Building blocks used below:
#   - BatchNorm1d / LayerNorm for feature normalization
#   - Dropout for regularization
#   - GELU activations
#   - Multi-head self-attention ("Attention Is All You Need")
#   - A learnable "class token" (as in ViT / CaiT) that attends over all
#     particle tokens to produce a single, permutation-invariant jet-level
#     representation used for the final classification
#
# NOTE: unlike the full Particle Transformer of Qu et al. (2022), this
# implementation does NOT include the paper's pairwise "particle
# interaction" features (the U matrix added inside P-MHA before the
# softmax). The authors explicitly note they attempted to reproduce it but
# found the required pieces underspecified in the paper, and left it out:
#
#   P-MHA(Q, K, V) = SoftMax(QK^T / sqrt(d_k) + U) V      <- U intentionally omitted here
#
# What remains is effectively the attention-only skeleton of ParT: particle
# embedding -> stacked self-attention ("particle attention") blocks ->
# class-attention pooling -> MLP classifier head. No ad-hoc positional
# encoding is added, since jet constituents are permutation-invariant
# (a point cloud, not a sequence with inherent order).

class Embed(nn.Module):
    """Projects raw per-particle features into the model's embedding space.

    Normalizes input (BatchNorm1d) across particles/batch, then applies a
    stack of (LayerNorm -> Linear -> GELU) blocks, one per entry in `dims`.
    """

    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
        x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x).transpose(0, 1)
        # (batch, seq_len, embed_dim)


class P_MHA(nn.Module):
    """A single particle (self-)attention block.

    Pre-norm multi-head self-attention + residual, followed by a pre-norm
    GELU MLP + residual (standard Transformer encoder block), as described
    for the ParT paper's particle-attention blocks minus the interaction
    bias U (see module-level note above).
    """

    def __init__(self, input_size, num_heads, linear_size, attn_dropout=0.1,
                 dropout=0.1, activation='gelu'):
        super().__init__()
        self.LN1 = nn.LayerNorm(input_size)
        self.MHA = nn.MultiheadAttention(input_size, num_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.LN2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, linear_size),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_size, input_size),
        )

    def forward(self, x, mask=None):
        # Change to (batch_size, seq_len, input_size)

        # Layer normalization at the start
        x_norm = self.LN1(x)
        # (batch, seq_len, input_size)

        # Multihead Attention
        attn_output, _ = self.MHA(x_norm, x_norm, x_norm, key_padding_mask=mask)
        # (batch, seq_len, embed_dim)

        output = self.LN2(attn_output)

        # Sum the initial input and the output of the last LayerNorm
        output_sum = x + output

        linear_out = self.linear(output_sum)

        return linear_out + output_sum


class L_blocks_P_MHA(P_MHA):
    """Stacks `L_blocks` particle-attention blocks, as in the paper's L=8 baseline.

    NOTE (kept as originally written): `mask` is accepted as an argument but
    each internal call below hardcodes `mask=None`, so the padding mask is
    not actually applied inside the stacked particle-attention blocks.
    """

    def forward(self, x, L_blocks=8, mask=None):
        x_i = x
        for i in range(L_blocks):
            new_x = super().forward(x_i, mask=None)
            x_i = new_x
        return new_x


class ClassAttentionBlock(nn.Module):
    """Class-attention block (as in CaiT): a learnable "class token" attends
    over the particle tokens (concatenated with itself) to aggregate a
    single, global, permutation-invariant jet-level representation.
    """

    def __init__(self, input_size, num_heads, linear_size, activation="gelu"):
        super().__init__()
        self.LN1 = nn.LayerNorm(input_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, input_size),
                                         requires_grad=True)  # Learnable class token
        self.mha = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.LN2 = nn.LayerNorm(input_size)
        self.linear = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, linear_size),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(linear_size, input_size),
        )

    def forward(self, x, x_cls=None, mask=None):
        # Change to (batch_size, seq_len, input_size)

        # Expand class token to match the batch size and sequence length
        if x_cls is None:
            class_token = self.class_token.expand(x.size(0), -1, -1)
        else:
            class_token = x_cls
        # Concatenate the class token with the sequence
        x_with_token = torch.cat((class_token, x), dim=1)

        x_with_token = self.LN1(x_with_token)

        # Calculate attention using Multihead Attention: the class token is
        # the query, the (class token + particles) sequence is key/value.
        attn_output, _ = self.mha(class_token, x_with_token, x_with_token,
                                   key_padding_mask=mask)

        output = self.LN2(attn_output)

        output_sum = class_token + output

        linear_out = self.linear(output_sum)
        print(x_with_token.shape, attn_output.shape, output.shape)  # leftover debug print (kept as-is)
        # Return attention output
        return linear_out + output_sum


class L_blocks_ClassAttentionBlock(ClassAttentionBlock):
    """Stacks `L_blocks` class-attention blocks (the paper's baseline uses 2)."""

    def forward(self, x, L_blocks=2, x_cls=None, mask=None):
        x_i_cls = None
        for i in range(L_blocks):
            new_x_cls = super().forward(x, x_cls=x_i_cls, mask=None)
            x_i_cls = new_x_cls
        return new_x_cls


class Final(nn.Module):
    """Classification head: LayerNorm on the pooled class token, followed by
    an MLP down to `num_classes` logits (softmax optionally applied for
    inference, `for_inf=True`, otherwise raw logits are returned for use
    with `F.cross_entropy`).
    """

    def __init__(self, input_size, dims, for_inf=False):
        super().__init__()
        self.for_inference = for_inf
        self.norm = nn.LayerNorm(input_size)
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.Linear(input_size, dim),
            ])
            input_size = dim
        self.lin = nn.Sequential(*module_list)

    def forward(self, x):
        out_norm = self.norm(x).squeeze(1)
        output = self.lin(out_norm)

        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output


class ParT(nn.Module):
    """Full Particle Transformer classifier: Embed -> stacked particle
    self-attention -> class-attention pooling -> MLP head.

    All hyperparameters (block widths, number of heads, number of layers,
    dropout, ...) are exposed as constructor arguments so the architecture
    can be tuned without touching the block implementations.
    """

    def __init__(self, input_dim_emb,
                 dims_emb,

                 input_size_pmha,
                 num_heads_pmha,
                 linear_size_pmha,

                 input_size_cls,
                 num_heads_cls,
                 linear_size_cls,

                 input_size_final,
                 dims_final,

                 normalize_input_emb=True,
                 attn_dropout_pmha=0.1,
                 dropout_pmha=0.1,
                 activation='gelu',
                 for_inf_final=False):

        super().__init__()

        self.emb = Embed(input_dim_emb, dims_emb, normalize_input=normalize_input_emb)
        self.pmha = L_blocks_P_MHA(input_size_pmha, num_heads_pmha,
                                    linear_size_pmha, attn_dropout=attn_dropout_pmha,
                                    dropout=dropout_pmha, activation='gelu')
        self.cls = L_blocks_ClassAttentionBlock(input_size_cls,
                                                 num_heads_cls, linear_size_cls)
        self.final = Final(input_size_final, dims_final, for_inf=for_inf_final)

    def forward(self, x, mask=None, x_cls=None):
        emb_out = self.emb(x)
        pmha_out = self.pmha(emb_out, mask=mask)
        cls_out = self.cls(pmha_out, mask=mask, x_cls=x_cls)
        output_sm = self.final(cls_out)

        return output_sm


# ==========================================================================
# 7. Hyperparameters
# ==========================================================================
# Taken from the ParT paper's baseline configuration (L=8 particle-attention
# blocks, 2 class-attention blocks, embedding dim d=128 via a 3-layer MLP
# with (128, 512, 128) nodes, 8 attention heads, dropout 0.1 on particle
# attention blocks, cross-entropy loss, RAdam + Lookahead optimizer,
# batch size 512):
#
#   "The baseline ParT model has a total of L = 8 particle attention blocks
#   and 2 class attention blocks. It uses a particle embedding of dimension
#   d = 128, encoded from the input particle features using a 3-layer MLP
#   with (128, 512, 128) nodes each layer with GELU nonlinearity [...] The
#   P-MHA (MHA) in the particle (class) attention blocks all have 8 heads
#   [...] We use a dropout of 0.1 for all particle attention blocks, and no
#   dropout for the class attention block."
#
#   "The training is performed on the full training set of 100M jets. We
#   employ the Lookahead optimizer (Zhang et al., 2019) with k=6 and
#   alpha=0.5 to minimize the cross-entropy loss, and the inner optimizer is
#   RAdam (Liu et al., 2020) with beta1=0.95, beta2=0.999, eps=1e-5. A batch
#   size of 512 and an initial learning rate (LR) of 0.001 are used."

TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

EPOCHS = 15

MAX_GRAD_NORM = 10

log_dir = "/content/drive/MyDrive/JetClass/logs"

input_dim_emb = 17
dims_emb = [128, 512, 128]

input_size_pmha = 128
num_heads_pmha = 8
linear_size_pmha = 128

input_size_cls = 128
num_heads_cls = 8
linear_size_cls = 128

input_size_final = 128
dims_final = [100, 50, 10]  # 10 output classes


# ==========================================================================
# 8. Sanity-check forward pass through each block
# ==========================================================================
# NOTE on fidelity: the following shape-probing cells were run out of order
# in the original interactive Colab session -- this first probe reuses
# `out0` / `cls` which are only produced by the model-instantiation cell
# further below. Kept in the original relative order rather than silently
# reordered; run the "model0 = Embed(...)" block below first if executing
# this file top-to-bottom end-to-end.
#
# ln = nn.LayerNorm(128)
# b = nn.BatchNorm1d(128)
# ln_o = ln(out0)
# ln_b = b(out0)
# print(ln_o.shape)
# mha = nn.MultiheadAttention(128, 8, batch_first=True)
# mha_o, _ = mha(cls, ln_o, ln_o)
# print(mha_o.shape)
# ln2 = ln(mha_o)
# print(ln2.shape)
# lin = nn.Linear(128, 100)
# lin_o = lin(ln2)
# print(lin_o.shape)
#
# class_token = nn.Parameter(torch.zeros(1, 1, 128), requires_grad=True)
# cls = class_token.expand(X_train[0:100].size(0), -1, -1)
# x_with_token = torch.cat((cls, out1[0:100]), dim=1)
# ln_cls = ln(x_with_token)
# mha_cls, _ = mha(cls, ln_cls, ln_cls)
# ln_cls.shape            # -> torch.Size([100, 129, 128])

model0 = Embed(input_dim_emb, dims_emb)
out0 = model0(X_train[0:100])
print(out0.shape)

model1 = L_blocks_P_MHA(input_size_pmha, num_heads_pmha, linear_size_pmha)
out1 = model1(out0, mask=train_mask[0:100])
print(out1.shape)

model2 = L_blocks_ClassAttentionBlock(input_size_cls, num_heads_cls, linear_size_cls)
out2 = model2(out1, mask=train_mask[0:100])
print(out2.shape)

model3 = Final(input_size_final, dims_final)
out3 = model3(out2)
print(out3.shape)

model = ParT(input_dim_emb=input_dim_emb,
             dims_emb=dims_emb,

             input_size_pmha=input_size_pmha,
             num_heads_pmha=num_heads_pmha,
             linear_size_pmha=linear_size_pmha,

             input_size_cls=input_size_cls,
             num_heads_cls=num_heads_cls,
             linear_size_cls=linear_size_cls,

             input_size_final=input_size_final,
             dims_final=dims_final)


# ==========================================================================
# 9. Training and testing (PyTorch Lightning)
# ==========================================================================
# In order to implement the phases of training, validation and testing we
# used PyTorch Lightning. The training/validation steps compute loss,
# accuracy and the (macro, one-vs-one) ROC-AUC, logging them per step/epoch.
# The test step additionally computes, per signal class, the background
# rejection (1 / false-positive-rate) at the signal-efficiency threshold
# used by the ParT paper for that class. A TensorBoard logger is used to
# track and visualize all metrics during training.
#
# Metrics (see README for the full explanation):
#   - Accuracy and macro AUC-ROC: overall multi-class discrimination.
#   - Background rejection at X% signal efficiency:
#         Rej_X% = 1 / FPR  at TPR = X%
#     an HEP-specific metric measuring how well the background (QCD) class
#     is rejected at a fixed true-positive rate for each signal class.

import pytorch_lightning as pl
from torch.optim import RAdam
from sklearn.metrics import roc_auc_score
from timm.optim.lookahead import Lookahead
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score
from torchmetrics.classification import MulticlassConfusionMatrix


class LightiningTrainVal(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.logg = TensorBoardLogger(log_dir, name="lightning_logs")

    def forward(self, x, mask=None):
        return self.model(x, mask=mask)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch  # Preprocess batch data

        # Forward pass
        logits = self.forward(x, mask)

        # Calculate loss
        loss = F.cross_entropy(logits, y.float())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        # Calculate accuracy
        logits_argmax = torch.argmax(logits, dim=1)
        y_argmax = torch.argmax(y, dim=1)
        acc = (logits_argmax == y_argmax).float().mean()
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)

        # Calculate AUC ROC
        logits_cpu = logits.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        logits_softmax = torch.softmax(torch.from_numpy(logits_cpu), dim=1).cpu().numpy()
        auc = roc_auc_score(y_cpu.flatten(), logits_softmax.flatten(), average="macro", multi_class="ovo")
        self.log('train_auc', auc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch  # Assuming mask is included in your dataset

        logits = self.forward(x, mask).float()
        loss = F.cross_entropy(logits, y.float())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)  # Logging the validation loss

        # Calculate training accuracy
        logits_argmax = torch.argmax(logits, dim=1)
        y_argmax = torch.argmax(y, dim=1)
        acc = (logits_argmax == y_argmax).float().mean()
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

        # Calculate AUC score
        logits_cpu = logits.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()

        probabilities = torch.softmax(torch.from_numpy(logits_cpu), dim=1).numpy()

        roc = roc_auc_score(y_cpu.flatten(), probabilities.flatten(), average="macro", multi_class="ovo")
        self.log('val_auc', roc, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch

        y_true_argm = torch.argmax(y, dim=1)
        logits = self.forward(x, mask)
        logits_argm = torch.argmax(logits, dim=1)
        device = torch.cuda.current_device()

        metric = MulticlassConfusionMatrix(num_classes=10).to(device)
        cm = metric(logits, y_true_argm)

        label_list = ['Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Tbqq', 'Tbl', 'Zqq', 'Wqq']
        signal_efficiency_thresholds = [0.5, 0.5, 0.5, 0.5, 0.99, 0.5, 0.995, 0.5, 0.5]

        for i in range(9):
            signal_class = i
            background_class = 9
            # Get confusion matrix values for signal and background classes
            tp = cm[signal_class][signal_class]
            fp = cm[background_class][signal_class]
            tn = cm[background_class][background_class]
            fn = cm[signal_class][background_class]

            # Calculate False Positive Rate (FPR)
            fpr = fp / (fp + tn)

            # Calculate Signal Efficiency (SE)
            se = tp / (tp + fn)

            # Calculate Background Rejection at the target Signal Efficiency
            signal_efficiency_threshold = signal_efficiency_thresholds[i]
            if se >= signal_efficiency_threshold and torch.isnan(fpr) == False:
                background_rejection = 1 / fpr
                if torch.isinf(background_rejection) == False:
                    self.log(f'test_rejection_score_{label_list[i]}', background_rejection, on_epoch=True)
            else:
                background_rejection = 0  # Signal efficiency is below the desired threshold

        # Compute loss
        loss = F.cross_entropy(logits, y.float())

        # compute accuracy
        acc = (logits_argm == y_true_argm).float().mean()

        logits_cpu = logits.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()

        probabilities = torch.softmax(torch.from_numpy(logits_cpu), dim=1).numpy()

        roc = roc_auc_score(y_cpu.flatten(), probabilities.flatten(), average="macro", multi_class="ovo")

        self.log('test_loss', loss, prog_bar=True, on_epoch=True)  # Logging the test loss
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
        self.log('test_roc', roc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # RANGER optimizer = RAdam (inner) + Lookahead (outer), as used in
        # the ParT paper. RAdam rectifies the adaptive learning rate's
        # variance early in training; Lookahead keeps a slow set of weights
        # updated from k fast-optimizer steps, for more stable convergence.
        inn_optimizer = RAdam(self.parameters(), lr=1e-3, betas=(0.95, 0.999), eps=1e-05)
        optimizer = Lookahead(inn_optimizer, alpha=0.5, k=6)
        return optimizer

    # Other methods as needed for Lightning functionality


light_model = LightiningTrainVal(model)


# --- Data loaders ---------------------------------------------------------
# Shuffling training data helps prevent any unintentional bias that could
# arise from the ordering of samples. Validation data is NOT shuffled, so
# evaluation is performed on samples in their original order.
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Wraps (features, labels, mask) arrays into a PyTorch Dataset so they
    can be consumed by a standard DataLoader.
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.c[idx]
        # return {"a": self.a[idx], "b": self.b[idx], "c": self.c[idx]}


train_data_ = CustomDataset(X_train, y_train, train_mask)
val_data_ = CustomDataset(X_val, y_val, val_mask)
test_data_ = CustomDataset(X_test, y_test, test_mask)

# Create the DataLoader objects for features, labels and mask
train_dataloader = torch.utils.data.DataLoader(train_data_, TRAIN_BATCH_SIZE, shuffle=True, num_workers=11)
val_dataloader = torch.utils.data.DataLoader(val_data_, VALID_BATCH_SIZE, shuffle=False, num_workers=7)
test_dataloader = torch.utils.data.DataLoader(test_data_, TEST_BATCH_SIZE, shuffle=False, num_workers=0)

torch.cuda.synchronize()
torch.cuda.empty_cache()

# !rm -rf /content/drive/MyDrive/JetClass/logs/lightning_logs
# %reload_ext tensorboard


# --- Trainer: checkpointing + early stopping ------------------------------
# At this point we have everything ready to create our trainer. We decided
# to include checkpoints and early stopping criteria as callbacks, the
# former to save and reuse the model in successive runs if necessary, and
# the latter to avoid overfitting.

### FINAL VERSION AFTER BEST TUNED PARAMETERS ###

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='/content/drive/MyDrive/JetClass/models',
    filename='best-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # Save the best model
    mode='min',    # 'min' or 'max' depending on the metric to monitor
)

# Define EarlyStopping callback
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss for early stopping
    patience=2,          # Number of epochs with no improvement after which training will be stopped
    min_delta=1e-2,
    mode='min',          # 'min' or 'max' depending on the metric to monitor
)

trainer = pl.Trainer(max_epochs=EPOCHS, default_root_dir=log_dir, logger=light_model.logg,
                      callbacks=[checkpoint_callback, early_stop_callback])
trainer.fit(light_model, train_dataloader, val_dataloader)

# At this point the original notebook launched:
#   %tensorboard --logdir=/content/drive/MyDrive/JetClass/logs/lightning_logs --port=6005
# to monitor training curves live. The extraction this script was
# reconstructed from is truncated shortly after this point (mid-first-epoch
# sanity checking / validation steps), so no final test_step metrics
# (test_acc / test_roc / test_rejection_score_*) are available in the
# source -- see README.md "Results" for details. To obtain them, run:
#
# trainer.test(light_model, test_dataloader)
