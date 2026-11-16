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
launch -- was fully recovered and is faithfully reproduced below.

Because the extraction cuts off while the first training epoch is still in
its "Sanity Checking" / early validation steps, no final numeric results
(test accuracy, ROC-AUC, background-rejection scores) are present in the
source. They are therefore NOT reported here or in the README -- see the
"Results" section of the README for what is and isn't known, including a
small reviewer-run forward-pass smoke test (see `--mode smoke-test` below).

--------------------------------------------------------------------------
REVIEWER PASS (post-migration code-quality review)
--------------------------------------------------------------------------
The original migration deliberately preserved two authentic quirks "as
written" and only flagged them in comments. This later review pass fixes
real bugs, including those two, and restructures the Colab notebook cells
into a runnable script with a CLI. Summary of substantive changes (kept
here rather than only in the PR description, since a reader of the code is
exactly who needs to know what changed and why):

  1. `L_blocks_P_MHA.forward` and `L_blocks_ClassAttentionBlock.forward`
     both hardcoded `mask=None` on every internal block call, so the
     padding mask was never actually applied anywhere in the network. This
     is fixed below: the mask is threaded through both stacks. Doing this
     correctly also required fixing a latent, previously-inert bug: the
     raw `pf_mask` convention is 1 = real particle / 0 = padding, whereas
     `nn.MultiheadAttention`'s `key_padding_mask` convention is the
     opposite (True = ignore/pad). `ParT.forward` now converts between the
     two explicitly instead of passing the raw mask straight through
     (which, had the hardcoded `None`s simply been removed without this,
     would have silently masked out every *real* particle instead of the
     padding).
  2. The leftover debug `print(...)` in `ClassAttentionBlock.forward` is
     removed.
  3. The macro one-vs-one ROC-AUC computed in `training_step` /
     `validation_step` / `test_step` flattened the one-hot labels and the
     per-class probabilities into 1D arrays before calling
     `sklearn.metrics.roc_auc_score(..., multi_class="ovo")`. That is not
     the API `multi_class="ovo"` expects (integer class labels of shape
     `(N,)` and a 2D probability array of shape `(N, C)`); flattening both
     turns it into a different, not-meaningful computation. Fixed to pass
     the already-computed integer argmax labels and the unflattened
     per-class probabilities, via a small `_macro_ovo_auc` helper that also
     passes `labels=range(num_classes)` so a mini-batch that happens not to
     contain every class (common during Lightning's initial "sanity check"
     validation batches, or a small trailing batch) doesn't crash the run
     with `ValueError: Number of classes in y_true not equal to the number
     of columns in 'y_score'`. The original flattened computation never hit
     this particular error, but only because flattening one-hot labels
     degrades the calculation into a meaningless binary AUC that bypasses
     sklearn's multiclass validation -- not because it was correct.
  4. `test_step` called `torch.cuda.current_device()` unconditionally,
     which raises on any machine without CUDA (e.g. CPU or Apple Silicon /
     MPS). Fixed to use the Lightning module's own `.device`.
  5. The Colab-specific `drive.mount(...)` call and hardcoded
     `/content/drive/...` paths are replaced with CLI arguments (with the
     same directory *layout* by default), and the notebook-style top-level
     statements (which executed on import / top-to-bottom cell execution)
     are reorganized into functions invoked from `main()`, so the file is
     an actual runnable script rather than disconnected notebook cells.
  6. Unused imports carried over from the notebook (torchvision, PIL,
     matplotlib/IPython display helpers, Axes3D, several unused stdlib
     modules) are removed.

Everything else -- the feature engineering, model architecture, training
hyperparameters and overall Lightning pipeline -- is unchanged from the
reconstructed original.
"""

import argparse
import glob
import gzip
import os
import pickle

import awkward as ak
import numpy as np
import pytorch_lightning as pl
import requests
import torch
import uproot
import vector
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from timm.optim.lookahead import Lookahead
from torch import nn
from torch.nn import functional as F
from torch.optim import RAdam
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassConfusionMatrix
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

vector.register_awkward()


# ==========================================================================
# 1. Download & explore the JetClass example dataset
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
        for chunk in resp.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            bar.update(size)


def download_and_explore_example(data_dir):
    """Downloads the small, official 100k-jet JetClass example file (if not
    already present) and prints its ROOT-tree schema -- useful purely to
    sanity-check the dataset layout before building the full training set.
    """
    os.makedirs(data_dir, exist_ok=True)
    example_path = os.path.join(data_dir, 'JetClass_example_100k.root')
    if not os.path.exists(example_path):
        _download(
            'https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root',
            example_path,
        )

    tree = uproot.open(example_path)['tree']
    tree.show()  # branch names / types

    table = tree.arrays()
    # Construct a Lorentz 4-vector from (px, py, pz, energy): the standard
    # way of characterising a particle's kinematics in HEP analyses.
    p4 = vector.zip({
        'px': table['part_px'],
        'py': table['part_py'],
        'pz': table['part_pz'],
        'energy': table['part_energy'],
    })
    print(f"Loaded {len(table)} jets from {example_path}")
    print(f"Example derived kinematics (first jet): pt={p4.pt[0][:5]}, "
          f"eta={p4.eta[0][:5]}, phi={p4.phi[0][:5]}")


# ==========================================================================
# 2. Feature engineering: padding, clipping, and building model inputs
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


def build_features_and_labels(a, max_particles=128, transform_features=True):
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
      - Pad every per-particle feature to a fixed `max_particles` constituents
        per jet and stack them into `pf_points`, `pf_features`, `pf_vectors`
        and `pf_mask` arrays.
      - Stack the 10 one-hot class labels into a single `label` array.
    """
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
            # Kinematics (measured in GeV): pt, energy and their jet-relative
            # log versions, plus the angular distance to the jet axis.
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',

            # 5-class one-hot encoding of the particle identity (+-1),
            # as determined by the detector's particle-flow reconstruction.
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',

            # Values and errors of the transverse and longitudinal impact
            # parameters of the particle trajectories, in mm. Only defined
            # for charged particles; neutral particles get 0.
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',

            # Pseudorapidity / azimuthal-angle difference between the
            # particle and the jet axis.
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
        out[k] = np.stack([_pad(a[n], maxlen=max_particles).to_numpy() for n in names], axis=1)

    label_list = ['label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql',
                  'label_Tbqq', 'label_Tbl', 'label_Zqq', 'label_Wqq', 'label_QCD']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out


# ==========================================================================
# 3. Building the full training dataset (~1M jets) from raw ROOT files
# ==========================================================================
# The full JetClass training partition is 100M jets, split across many
# per-class ROOT files (see https://github.com/jet-universe/particle_transformer
# `get_datasets.py` for how to download it). For this project, to keep
# things tractable on Colab while remaining representative of the full
# dataset, a stratified subsample was taken: one file in every `stride`
# (sorted, so the subsample is reproducible), evenly covering all 10
# classes.
def build_training_dataset(root_glob_pattern, cache_path, stride=30, max_particles=128):
    """Globs ROOT files matching `root_glob_pattern`, subsamples every
    `stride`-th file (sorted for reproducibility), builds the
    (features, labels, mask, ...) arrays via `build_features_and_labels`,
    and caches the result as a gzip-compressed pickle at `cache_path`.

    Requires the raw JetClass ROOT files to already be present locally
    (see README.md for how to obtain them) -- this function does not
    download the full (multi-hundred-GB) dataset itself.
    """
    file_list = sorted(glob.glob(root_glob_pattern))
    if not file_list:
        raise FileNotFoundError(
            f"No ROOT files matched '{root_glob_pattern}'. Download the JetClass "
            "training partition first (see README.md 'Getting the dataset')."
        )
    file_list = file_list[::stride]
    print(f"Building training dataset from {len(file_list)} files "
          f"(stride={stride}) matching '{root_glob_pattern}'")

    trees = ak.concatenate([
        uproot.open(file)["tree"].arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])
        for file in file_list
    ])

    dataset = build_features_and_labels(trees, max_particles=max_particles)

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Wrote {cache_path} ({sum(v.nbytes for v in dataset.values()) / 1e6:.1f} MB uncompressed)")
    return dataset


def load_dataset_splits(cache_path, seed=42):
    """Loads the cached (features, labels, mask) dataset built by
    `build_training_dataset` and splits it 64% train / 16% val / 20% test
    (an 80/20 split, then an 80/20 split of the training portion).
    """
    with gzip.open(cache_path, 'rb') as f:
        dataset = pickle.load(f)

    labels = torch.as_tensor(dataset["label"], dtype=torch.long)
    features = torch.from_numpy(dataset["pf_features"])
    mask = torch.from_numpy(dataset["pf_mask"])

    X_train, X_test, y_train, y_test, train_mask, test_mask = train_test_split(
        features, labels, mask, test_size=0.2, train_size=0.8, random_state=seed)
    X_train, X_val, y_train, y_val, train_mask, val_mask = train_test_split(
        X_train, y_train, train_mask, test_size=0.2, train_size=0.8, random_state=seed)

    assert train_mask.shape[0] == X_train.shape[0] == y_train.shape[0]

    return {
        "train": (X_train, y_train, train_mask),
        "val": (X_val, y_val, val_mask),
        "test": (X_test, y_test, test_mask),
    }


# ==========================================================================
# 4. Model architecture: Particle Transformer (ParT)
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
        # x: (batch_size, seq_len, input_size)
        # `mask` is a boolean key_padding_mask (True = padding, ignored by
        # attention), shape (batch_size, seq_len) -- see ParT.forward for
        # where it's derived from the raw per-particle pf_mask.

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
    """Stacks `L_blocks` particle-attention blocks, as in the paper's L=8 baseline."""

    def forward(self, x, L_blocks=8, mask=None):
        # Reviewer fix: this used to hardcode `mask=None` on every block
        # call below, so the padding mask was silently never applied here.
        # `mask` is now threaded through every stacked block.
        x_i = x
        for i in range(L_blocks):
            new_x = super().forward(x_i, mask=mask)
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
        # x: (batch_size, seq_len, input_size)
        # `mask`: boolean key_padding_mask over the seq_len particle tokens
        # (True = padding). The class token itself is never padding, so a
        # `False` column is prepended for it before it's passed to `mha`.

        # Expand class token to match the batch size and sequence length
        if x_cls is None:
            class_token = self.class_token.expand(x.size(0), -1, -1)
        else:
            class_token = x_cls
        # Concatenate the class token with the sequence
        x_with_token = torch.cat((class_token, x), dim=1)

        x_with_token = self.LN1(x_with_token)

        if mask is not None:
            cls_token_mask = torch.zeros(mask.size(0), 1, dtype=torch.bool, device=mask.device)
            mask_with_token = torch.cat((cls_token_mask, mask), dim=1)
        else:
            mask_with_token = None

        # Calculate attention using Multihead Attention: the class token is
        # the query, the (class token + particles) sequence is key/value.
        attn_output, _ = self.mha(class_token, x_with_token, x_with_token,
                                   key_padding_mask=mask_with_token)

        output = self.LN2(attn_output)

        output_sum = class_token + output

        linear_out = self.linear(output_sum)
        # Return attention output
        return linear_out + output_sum


class L_blocks_ClassAttentionBlock(ClassAttentionBlock):
    """Stacks `L_blocks` class-attention blocks (the paper's baseline uses 2)."""

    def forward(self, x, L_blocks=2, x_cls=None, mask=None):
        # Reviewer fix: same hardcoded-`mask=None` issue as L_blocks_P_MHA,
        # fixed the same way.
        x_i_cls = None
        for i in range(L_blocks):
            new_x_cls = super().forward(x, x_cls=x_i_cls, mask=mask)
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
                 for_inf_final=False,
                 num_pmha_blocks=8,
                 num_cls_blocks=2):

        super().__init__()

        self.emb = Embed(input_dim_emb, dims_emb, normalize_input=normalize_input_emb)
        self.pmha = L_blocks_P_MHA(input_size_pmha, num_heads_pmha,
                                    linear_size_pmha, attn_dropout=attn_dropout_pmha,
                                    dropout=dropout_pmha, activation='gelu')
        self.cls = L_blocks_ClassAttentionBlock(input_size_cls,
                                                 num_heads_cls, linear_size_cls)
        self.final = Final(input_size_final, dims_final, for_inf=for_inf_final)
        # Exposed so the depth can be tuned without touching the block
        # classes; defaults (8, 2) match the paper's baseline, as before.
        self.num_pmha_blocks = num_pmha_blocks
        self.num_cls_blocks = num_cls_blocks

    def forward(self, x, mask=None, x_cls=None):
        # `mask`: the raw per-particle pf_mask, shape (batch, 1, seq_len),
        # with 1 = real particle / 0 = padding (as produced by
        # `build_features_and_labels`). `nn.MultiheadAttention` expects the
        # opposite convention for `key_padding_mask` (True = ignore), so it
        # is inverted (and squeezed to (batch, seq_len)) here, once, before
        # being passed down to the attention blocks.
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask.reshape(mask.size(0), -1) == 0

        emb_out = self.emb(x)
        pmha_out = self.pmha(emb_out, L_blocks=self.num_pmha_blocks, mask=key_padding_mask)
        cls_out = self.cls(pmha_out, L_blocks=self.num_cls_blocks, mask=key_padding_mask, x_cls=x_cls)
        output = self.final(cls_out)

        return output


# ==========================================================================
# 5. Lightning module: training / validation / test steps
# ==========================================================================
# The training/validation steps compute loss, accuracy and the (macro,
# one-vs-one) ROC-AUC, logging them per step/epoch. The test step
# additionally computes, per signal class, the background rejection
# (1 / false-positive-rate) at the signal-efficiency threshold used by the
# ParT paper for that class. A TensorBoard logger tracks all metrics.
#
# Metrics (see README for the full explanation):
#   - Accuracy and macro AUC-ROC: overall multi-class discrimination.
#   - Background rejection at X% signal efficiency:
#         Rej_X% = 1 / FPR  at TPR = X%
#     an HEP-specific metric measuring how well the background (QCD) class
#     is rejected at a fixed true-positive rate for each signal class.
def _macro_ovo_auc(y_true, probabilities, num_classes):
    """Macro one-vs-one ROC-AUC, robust to a batch not containing every
    class (e.g. the first sanity-check mini-batch, or a small/uneven last
    batch): `labels=range(num_classes)` tells sklearn the score matrix has
    `num_classes` columns regardless of how many are actually present in
    `y_true` for this particular batch, instead of raising
    `ValueError: Number of classes in y_true not equal to the number of
    columns in 'y_score'`. If a batch is so degenerate it has only one
    class present at all, this returns NaN (mathematically undefined AUC)
    rather than crashing.
    """
    return roc_auc_score(y_true, probabilities, average="macro", multi_class="ovo",
                          labels=list(range(num_classes)))


class JetTaggingLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper around `ParT` implementing the train/val/test
    loops, metric logging and the Ranger-equivalent (RAdam + Lookahead)
    optimizer used in the ParT paper.
    """

    def __init__(self, model, log_dir, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.tb_logger = TensorBoardLogger(log_dir, name="lightning_logs")

    def forward(self, x, mask=None):
        return self.model(x, mask=mask)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch

        logits = self.forward(x, mask)
        loss = F.cross_entropy(logits, y.float())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        logits_argmax = torch.argmax(logits, dim=1)
        y_argmax = torch.argmax(y, dim=1)
        acc = (logits_argmax == y_argmax).float().mean()
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)

        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        auc = _macro_ovo_auc(y_argmax.detach().cpu().numpy(), probabilities, logits.shape[1])
        self.log('train_auc', auc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch

        logits = self.forward(x, mask).float()
        loss = F.cross_entropy(logits, y.float())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)

        logits_argmax = torch.argmax(logits, dim=1)
        y_argmax = torch.argmax(y, dim=1)
        acc = (logits_argmax == y_argmax).float().mean()
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        roc = _macro_ovo_auc(y_argmax.detach().cpu().numpy(), probabilities, logits.shape[1])
        self.log('val_auc', roc, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch

        y_true_argm = torch.argmax(y, dim=1)
        logits = self.forward(x, mask)
        logits_argm = torch.argmax(logits, dim=1)

        metric = MulticlassConfusionMatrix(num_classes=10).to(self.device)
        cm = metric(logits, y_true_argm)

        label_list = ['Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Tbqq', 'Tbl', 'Zqq', 'Wqq']
        signal_efficiency_thresholds = [0.5, 0.5, 0.5, 0.5, 0.99, 0.5, 0.995, 0.5, 0.5]

        for i in range(9):
            signal_class = i
            background_class = 9
            tp = cm[signal_class][signal_class]
            fp = cm[background_class][signal_class]
            tn = cm[background_class][background_class]
            fn = cm[signal_class][background_class]

            fpr = fp / (fp + tn)
            se = tp / (tp + fn)

            signal_efficiency_threshold = signal_efficiency_thresholds[i]
            if se >= signal_efficiency_threshold and not torch.isnan(fpr):
                background_rejection = 1 / fpr
                if not torch.isinf(background_rejection):
                    self.log(f'test_rejection_score_{label_list[i]}', background_rejection, on_epoch=True)

        loss = F.cross_entropy(logits, y.float())
        acc = (logits_argm == y_true_argm).float().mean()

        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
        roc = _macro_ovo_auc(y_true_argm.detach().cpu().numpy(), probabilities, logits.shape[1])

        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
        self.log('test_roc', roc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # RANGER optimizer = RAdam (inner) + Lookahead (outer), as used in
        # the ParT paper. RAdam rectifies the adaptive learning rate's
        # variance early in training; Lookahead keeps a slow set of weights
        # updated from k fast-optimizer steps, for more stable convergence.
        inn_optimizer = RAdam(self.parameters(), lr=self.lr, betas=(0.95, 0.999), eps=1e-05)
        optimizer = Lookahead(inn_optimizer, alpha=0.5, k=6)
        return optimizer


class JetTensorDataset(Dataset):
    """Wraps (features, labels, mask) tensors into a PyTorch Dataset so they
    can be consumed by a standard DataLoader.
    """

    def __init__(self, features, labels, mask):
        self.features = features
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.mask[idx]


# ==========================================================================
# 6. Model construction helper + forward-pass smoke test
# ==========================================================================
def build_model(input_dim=17, num_classes=10):
    """Constructs a `ParT` model with the ParT paper's baseline
    hyperparameters (d=128 embedding, L=8 particle-attention blocks, L=2
    class-attention blocks, 8 heads, dropout 0.1), as used throughout this
    project. Unlike CNNs/RNNs, attention has no fixed sequence-length
    dependence baked into its weights, so the number of particles per jet
    (`--max-particles`) does not affect the model's parameter shapes.
    """
    return ParT(
        input_dim_emb=input_dim,
        dims_emb=[128, 512, 128],

        input_size_pmha=128,
        num_heads_pmha=8,
        linear_size_pmha=128,

        input_size_cls=128,
        num_heads_cls=8,
        linear_size_cls=128,

        input_size_final=128,
        dims_final=[100, 50, num_classes],
    )


def run_smoke_test(max_particles=128, input_dim=17, num_classes=10, batch_size=4, device=None):
    """Instantiates the model and runs a forward pass on random dummy
    tensors of the right shape, with no dataset required. This does not
    validate numerical correctness against any reference, but does verify
    that the model builds and a forward pass through every block (Embed,
    particle attention, class attention, classification head) runs without
    shape/dtype errors -- useful as a fast, dependency-light check that the
    reconstructed/fixed code actually works end-to-end.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(input_dim=input_dim, num_classes=num_classes).to(device)
    model.eval()

    x = torch.randn(batch_size, input_dim, max_particles, device=device)
    # Random 0/1 mask with ~70% real particles, 30% padding -- shape and
    # convention matching the real pf_mask (1 = real particle, 0 = pad).
    mask = (torch.rand(batch_size, 1, max_particles, device=device) > 0.3).float()

    with torch.no_grad():
        logits = model(x, mask=mask)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke-test] device={device}  n_params={n_params:,}")
    print(f"[smoke-test] input x={tuple(x.shape)}  mask={tuple(mask.shape)}")
    print(f"[smoke-test] output logits={tuple(logits.shape)}")

    assert logits.shape == (batch_size, num_classes), (
        f"expected output shape {(batch_size, num_classes)}, got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "model produced non-finite logits"

    print("[smoke-test] OK: model instantiates and a forward pass "
          "produces finite logits of the expected shape.")
    return logits


# ==========================================================================
# 7. CLI entry point
# ==========================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Particle Transformer (ParT) for JetClass jet tagging.")
    parser.add_argument(
        '--mode', choices=['smoke-test', 'download-example', 'build-dataset', 'train'],
        default='smoke-test',
        help="smoke-test (default): instantiate the model and run a forward pass on "
             "random dummy tensors, no dataset required. download-example: fetch and "
             "inspect the small 100k-jet JetClass example file. build-dataset: subsample "
             "and cache a training set from local JetClass ROOT files. train: train on a "
             "previously built/cached dataset.")

    parser.add_argument('--data-dir', default='data/JetClass',
                         help="Root directory for JetClass data (default: data/JetClass).")
    parser.add_argument('--root-glob-pattern', default=None,
                         help="Glob pattern for raw per-class training ROOT files "
                              "(default: '<data-dir>/train_100M/*.root').")
    parser.add_argument('--cache-path', default=None,
                         help="Path to the cached (gzip-pickled) training dataset "
                              "(default: '<data-dir>/JetClass_subset.pkl.gz').")
    parser.add_argument('--stride', type=int, default=30,
                         help="Use every Nth sorted ROOT file when building the dataset "
                              "(default: 30, i.e. ~1/30th of the 100M-jet training set).")
    parser.add_argument('--max-particles', type=int, default=128,
                         help="Zero-pad/truncate each jet to this many constituents (default: 128).")

    parser.add_argument('--log-dir', default='logs', help="TensorBoard log directory.")
    parser.add_argument('--checkpoint-dir', default='checkpoints', help="Model checkpoint directory.")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--accelerator', default='auto',
                         help="Passed to pytorch_lightning.Trainer (e.g. 'cpu', 'gpu', 'mps', 'auto').")
    parser.add_argument('--run-test', action='store_true',
                         help="After training, also run trainer.test() on the held-out test split.")
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    root_glob_pattern = args.root_glob_pattern or os.path.join(args.data_dir, 'train_100M', '*.root')
    cache_path = args.cache_path or os.path.join(args.data_dir, 'JetClass_subset.pkl.gz')

    if args.mode == 'smoke-test':
        run_smoke_test(max_particles=args.max_particles)
        return

    if args.mode == 'download-example':
        download_and_explore_example(args.data_dir)
        return

    if args.mode == 'build-dataset':
        build_training_dataset(root_glob_pattern, cache_path,
                                 stride=args.stride, max_particles=args.max_particles)
        return

    # args.mode == 'train'
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached dataset found at '{cache_path}'. Run with --mode build-dataset "
            "first (see README.md 'Getting the dataset' for how to obtain the raw "
            "JetClass ROOT files)."
        )

    splits = load_dataset_splits(cache_path, seed=args.seed)
    X_train, y_train, train_mask = splits["train"]
    X_val, y_val, val_mask = splits["val"]

    train_loader = DataLoader(JetTensorDataset(X_train, y_train, train_mask),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(JetTensorDataset(X_val, y_val, val_mask),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = build_model()
    lightning_module = JetTaggingLightningModule(model, log_dir=args.log_dir, lr=args.lr)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_dir,
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=1e-2,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=args.log_dir,
        logger=lightning_module.tb_logger,
        accelerator=args.accelerator,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(lightning_module, train_loader, val_loader)

    if args.run_test:
        X_test, y_test, test_mask = splits["test"]
        test_loader = DataLoader(JetTensorDataset(X_test, y_test, test_mask),
                                  batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        trainer.test(lightning_module, test_loader)


if __name__ == "__main__":
    main()
