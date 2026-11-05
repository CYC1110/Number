# Validation record

## Checks actually performed

- 16 standard-library unittest tests passed (final cohort correction run: 2.917 seconds). Tests use small in-memory numeric arrays and temporary synthetic JSON metric fixtures; no stimulus image was generated and no visual backbone was instantiated/trained.
- All 23 package modules imported, including optional shader renderer code without creating an EGL context. All Python source files parsed successfully.
- 29 README CLI command lines were validated as help/plans, with `--execute` deliberately removed. All 290 configured state/run training/analysis plans and 13 figure plans were checked.
- Every training plan points to the same central checkpoint path used by its analysis plan. The four original conditions have 20 runs each; the other 13 conditions have five each (145 state/run combinations); escaped state names/path traversal are rejected.
- Privacy/path scan passed with no findings after removing preparation-time bytecode caches. No symlink, original-tree path, personal email, credential-like literal, notebook or binary research artifact is included. Heuristic results are recorded in `STATIC_AUDIT.json`; this is not a guarantee about future user-supplied data.
- Nine reviewed original primitive source files still matched their recorded hashes at the end of preparation. The original research tree was never a write target.

## What the tests establish

The tests verify exact P-edge assignment, stable cell-by-regime balancing and disjoint folds, Gaussian BC against its analytic value and symmetry, positive-definiteness failures, 90-degree orthogonal subspaces/rank-deficient unions, angle/BC invariance under common scalar rescaling, idempotent raw-origin ablation and the frozen-logit low-rank identity, balanced ANOVA quantity effects, decoder train/validation overlap guards, SimCLR/DINO losses on small embeddings, and translation/identity cases for FID. Cohort tests verify the exact original 20-run seed mapping, state-specific run validity, complete 20/five-network aggregation, checkpoint/metric seed mismatch and missing-record rejection, actual-n SEM, row-order-independent pairing, 20-to-five shared-seed pairing with explicit unmatched counts, equal-size but partially overlapping seed sets, and rejection of positional arrays, duplicate identities, nonfinite values and no-overlap pairing.

## What was not performed

No ImageNet or synthetic dataset loading, NP/single-color image generation, model/checkpoint creation, visual model training, full feature extraction, large readout/ablation/SVM evaluation, FID feature extraction, actual paper-value reproduction, final SVG rendering from research outputs, browser camera review, fresh dependency installation, or GitHub upload was performed. Optional hardware EGL was not tested. The dependency versions in `TEST_ENVIRONMENT.json` describe the available lightweight-test runtime, not a historical lock or validated clean installation.

Figure layouts were checked through code/import/plan inspection only. Full data-dependent commands intentionally fail when required inputs are absent. The single-color placement safeguards were inspected and syntax-checked, not exercised by generating an image. Final manuscript identity, external asset/data access, third-party rights and project license remain unresolved.

This package is a methodological reference. Passing the lightweight tests does not prove the scientific conclusions or numerical agreement with the original manuscript.
