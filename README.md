# Quantity Geometry — Reference Implementation

This repository explains and implements the analysis of numerosity (N) and proportion (P) representations in visual networks. It is prepared for methodological review, with **20 networks each for Untrained, Pretrained, Label-shuffled and DVD, and five for every other configured condition**. It is a reference implementation, not a reconstruction of the original execution environment, checkpoint cohorts, Illustrator artwork, or reported numerical results.

No training, dataset image generation, full feature extraction, or numerical reproduction of paper results was performed while preparing this repository. Only source checks, imports, CLI plans, and small in-memory numerical tests were run. The candidate coverage is Figure 1–6 and S1–S7 in manuscript directory `260925B`; its identity as the final manuscript has not been confirmed.

Author identification is withheld pending confirmation. The project license is **undecided**; no license or right to redistribute third-party datasets/assets is asserted. Nothing has been uploaded to GitHub.

## Installation

Use Python 3.10 or later and a compatible PyTorch/torchvision pair:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
iclr-repro --help
```

`requirements.txt` lists the dependencies actually imported by the core code. Its lower bounds are compatibility declarations, not an environment lock or a claim that every version combination was tested. Optional EGL shader generation uses `requirements-shaders.txt`. Optional upstream Clean-FID feature preparation uses `requirements-fid.txt`. Tests use standard-library `unittest`; `requirements-dev.txt` has no additional dependencies. Nothing downloads datasets or model weights automatically.

## Repository layout

```text
configs/                 condition-specific cohorts and NP stimulus configuration
src/iclr_repro/
  paths.py               central, checkout-contained path contracts
  datasets/              NP/single-color/synthetic generation and plans
  training.py            readable reference training algorithms
  pipeline.py            frozen feature extraction and analysis orchestration
  geometry.py            P7 N1/P2, held-out Gaussian overlap, raw ablation
  readouts.py            affine decoder, SVM, transfer, network-level statistics
  estimators/            preserved PCA/split/BC/ANOVA numerical primitives
  figures.py             new reference layouts and offline 3D viewer
  fid.py                 FID from supplied local frozen features
  audit.py               privacy/path/source scan
data/                    supplied/generated datasets; excluded from Git
checkpoints/             all training outputs; excluded from Git
results/                 analysis arrays, metrics, readouts; excluded from Git
figures/                 rendered reference figures; excluded from Git
docs/                    figure map, methodology, limitations, validation
tests/                   small-array tests, no image or network training
```

All data, checkpoints and outputs belong inside this checkout. There are no links to the original research tree, external drives, sibling virtual environments or external local Python modules. Resolved paths and dataset records that escape the checkout, including symlink targets, are rejected. `paths.py` is the single path authority. Move the whole folder freely; edit relative configuration rather than embedding a machine path.

## Conditions and seeds

`configs/protocol.json` explicitly assigns a cohort to every state. Untrained, Pretrained, Label-shuffled and DVD preserve the original **20 runs, `out_04`–`out_23`**: seeds 5962–5978 for `out_04`–`out_20`, then 5959, 5960 and 5961 for `out_21`–`out_23`. SimCLR, RotNet, DINO and the four synthetic diets use **five runs, `out_04`–`out_08`, seeds 5962–5966**. The four partial correct-label controls and both ResNet-50 controls retain the existing five-run reference configuration; their historical cohort sizes are not asserted. See [docs/COHORTS.md](docs/COHORTS.md) for the exact mapping and pairing evidence.

Evaluation uses seven P classes with edges `(1, 3, 5, 7, 9, 11)/12`, fixed N rank 1 and P rank 2, and split seed 1729. The N/P retained ranks are not selected independently per condition. There is no guarantee that they explain more than 90% variance in every condition. Untrained endpoints are epoch 0, RotNet ends at 30, and other configured endpoints are 90. Original variants such as 100-epoch SimCLR, other historical cohort variants, 105k synthetic pilots and continuation experiments are not silently pooled into these configured cohorts.

## Data preparation

ImageNet-1K is user supplied under `data/imagenet/train/<class>/...` and `data/imagenet/val/<class>/...`, with matching class-directory names. Reorganize the original flat validation set using its authorized annotations first; this repository does not guess labels. The reference trainer requires exactly 1,281,167 training images. DVD is a user supplied degraded visual diet under `data/dvd/<class>/...`; the historical developmental transformation is not approximated by an undocumented blur. No dataset is included or downloaded.

The following commands **print plans only**:

```bash
iclr-repro generate-np
iclr-repro generate-numerosity
iclr-repro generate-synthetic noise
iclr-repro generate-synthetic grating
iclr-repro generate-synthetic statistical
iclr-repro generate-synthetic shaders
```

Explicit future generation commands (not executed during preparation):

```bash
iclr-repro generate-np --execute
iclr-repro generate-numerosity --execute
iclr-repro generate-synthetic noise --execute
iclr-repro generate-synthetic grating --execute
iclr-repro generate-synthetic statistical --execute
iclr-repro generate-synthetic shaders --execute
```

NP uses the audited configuration: 227×227 grayscale images, N=0–30, 200 images per N per regime, constant size / color-specific total area / spatial extent controls, set-2 area 8000 and minimum dot area 113. Generation produces 18,600 images including 600 blanks. Analysis removes N=0, leaving 18,000 images and 495 `(N,W)` cells. Placement failures raise errors. The single-color numerosity source is separate: 16 N levels × 3 regimes × 50 images = 2,400 images. It is used for the legacy ANOVA/selective-unit analysis rather than treating NP images as its equivalent.

Noise and Grating preserve the original per-index generators. Statistical selects 1,281,167 supplied official Spectrum + Color + WMM RGB 128×128 PNGs from `data/assets/wmm_official/`; it does not substitute a homemade texture distribution. Shaders requires authorized local program files, a manifest and templates under `data/assets/shaders/`; see [data/assets/README.md](data/assets/README.md). Procedural Shader pixels depend on the graphics stack and are not claimed to replay exactly across machines.

## Training and checkpoint contract

The trainer implements supervised cross-entropy, fixed complete/partial label permutation, SimCLR NT-Xent, four-way rotation classification and DINO cross-view teacher targets with EMA/centering. The compact PIL backend, simplified DINO head/crop transform, optimizer defaults and schedules are reference choices, clearly distinguished from the original DALI recipes in [docs/METHODS.md](docs/METHODS.md). Synthetic visual diets use SimCLR. DVD uses supplied degraded images with supervised learning. Training creates every checkpoint inside this repository.

```bash
iclr-repro train pretrained --run out_04
iclr-repro train simclr --run out_04
```

These print plans. A future 20-run supervised job can be requested explicitly:

```bash
for index in {4..23}; do
  printf -v run "out_%02d" "$index"
  iclr-repro train pretrained --run "$run" --device cuda --execute
done
```

Use the run list assigned to each state in `configs/protocol.json`; five-run states accept only `out_04`–`out_08`. Replace `pretrained` with a configured state to inspect its plan. Use a compatible batch size via `--batch-size`. `untrained` saves an initialization only when `--execute` is given. The trainer refuses to overwrite existing checkpoints. It does not implement training resume; use a fresh checkout/run if an interrupted reference job must be repeated.

Checkpoint locations are identical for training and analysis:

```text
checkpoints/<state>/<run>/epoch_000.pth
checkpoints/<state>/<run>/epoch_090.pth
checkpoints/rotnet/<run>/epoch_030.pth
checkpoints/<state>/<run>/training_manifest.json
```

The checkpoint records its state, run, seed, epoch and architecture. Analysis rejects identity mismatches and missing convolution parameters. It never fills missing learned weights from a random initialization. DINO evaluation uses the EMA teacher encoder. Readout checkpoints belong to `results/`, separate from network training checkpoints.

## Analysis and execution order

1. Supply/generate the required datasets and local assets.
2. Train or supply review-format checkpoints for the full configured cohort of each condition.
3. Analyze each checkpoint independently, then aggregate its 20 or five networks.
4. Render a requested figure only after its required inputs exist.

```bash
iclr-repro analyze pretrained --run out_04 --epoch 90
iclr-repro analyze pretrained --run out_04 --epoch 90 --device cuda --execute
iclr-repro analyze untrained --run out_04 --epoch 0 --device cuda --execute
iclr-repro analyze pretrained --run out_04 --epoch 90 --single-unit --device cuda --execute
```

The first command prints the shared checkpoint/output paths. The `--execute` variants are intended future commands, not preparation-time validation. Repeat the relevant analysis for every run ID configured for that state and every state needed by a figure. Full feature extraction and decoder training can require substantial RAM/GPU resources; they were not run here.

Geometry uses full-data conditioned means for the descriptive principal angle. For BC, a three-fold cell-by-regime balanced split fits N1/P2 on two folds, estimates Ledoit–Wolf cell Gaussians on the held-out fold, computes all 122,265 unordered cell-pair coefficients, averages coefficients within the fold, then averages the three folds. A singular covariance makes the canonical value undefined; no ridge, valid-pair substitution, interpolation or finite-only cohort averaging is used.

Affine N/P/NP decoders use fold A for fitting, B for validation-only early stopping and C for testing. They have no hidden layer or test-fitted scaler. Cross-quantity ablation removes P2 for N readout or N1 for P readout at the raw origin, holding the fitted decoder fixed. Damage is the change in ordinal MAE. The default ablation basis is full-data/descriptive, matching the reviewed analysis definition; it is **not fully inductive** because test stimuli/labels enter basis fitting. `--basis-scope train` provides an explicit train-only sensitivity variant. See [docs/METHODS.md](docs/METHODS.md) for this limitation and other reference-vs-original differences.

```bash
iclr-repro analyze pretrained --run out_04 --epoch 90 --basis-scope train --execute
```

Use a fresh output tree for sensitivity analysis: existing outputs are not overwritten or mixed. The restricted pairwise SVM, incongruent readout and bidirectional cross-context transfer are implemented, with explicit matching/support rules. These rules are reference definitions rather than the original saved pair/decoder registries.

Each analysis writes:

```text
results/<state>/<run>/epoch_<NNN>/metrics.json
results/<state>/<run>/epoch_<NNN>/analysis_table.csv
results/<state>/<run>/epoch_<NNN>/split_balance.csv
results/<state>/<run>/epoch_<NNN>/bases.npz
results/<state>/<run>/epoch_<NNN>/geometry_fold_{0,1,2}.npz
results/<state>/<run>/epoch_<NNN>/test_predictions.npz
results/<state>/<run>/epoch_<NNN>/decoder_{N,P,NP}.pth
```

Provenance contains project-relative paths and hashes, not machine paths or author metadata. Full activation arrays are held in memory rather than published.

## Figure commands

```bash
iclr-repro plot F1
iclr-repro plot F3 --execute
iclr-repro plot S7 --execute
```

Every figure has `iclr-repro plot <ID> --execute`, for `F1`–`F6` and `S1`–`S7`. The default without `--execute` prints a plan. [docs/FIGURES.md](docs/FIGURES.md) maps every candidate figure to its data, checkpoints, estimators, required states, and exact commands. Outputs are new method layouts under `figures/<ID>.svg` plus JSON sidecars; S7 is a self-contained offline HTML viewer. Original hand-edited SVG layouts and human-behavior insets are not regenerated or claimed to be generated by these commands.

S6 uses 50,000 local frozen Clean-FID features per image distribution and supplied extractor provenance. See [data/fid_features/README.md](data/fid_features/README.md):

```bash
iclr-repro fid
iclr-repro fid --execute
iclr-repro plot S6 --execute
```

## Lightweight validation

```bash
python -m unittest discover -s tests -v
iclr-repro paths
iclr-repro audit
```

Tests cover imports, 20/five-run path identity, complete-cohort aggregation, actual-n SEM, explicit seed pairing and unequal-cohort guards, exact P edges, balanced/disjoint splits, analytic BC, angle/union degeneracy, scalar invariance, raw-origin ablation and frozen logits, balanced ANOVA, decoder split guards, small-array losses and analytic FID. They do not generate images, instantiate/train visual backbones or verify the reported paper numbers. [docs/VALIDATION.md](docs/VALIDATION.md) records actual checks and remaining limits.

The release excludes private notes, original reports/manuscripts, notebooks and their outputs, logs, cached activations, old checkpoints, binary figures, source archives and unneeded exploratory analyses. `.gitignore` retains dataset/checkpoint/output READMEs while excluding populated data, checkpoints, outputs, caches, environments and credential files. License/third-party asset authorization and final manuscript identity remain unresolved; this is local preparation, not a completed publication.

## Interactive 3D viewers and custom network responses

- [Network endpoint comparison](viewers/compare.html): 33 existing state/run endpoints, with the actual epoch displayed.
- [Epoch comparison and animation](viewers/epochs.html): pretrained `out_21` epochs 0–90, synchronized multi-network panels and GIF export.
- [Viewer guide and response template](docs/VIEWERS.md): upload arbitrary network responses for the exact published NP dataset, or load compact 3D projection JSON directly. Run `python -m iclr_repro.viewer serve` for raw NPZ uploads. Both generated HTML files also work offline with their embedded data.

The shipped historical P6 projections are identified separately from newly generated ICLR dataset measurements. No raw activation cache is bundled. Rebuild via `python -m iclr_repro.viewer build`; inputs are configured in `configs/viewers.json`.
