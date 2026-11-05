# Interactive N/P representations

Two self-contained HTML files are included. They contain the measured 3D projections and all rendering code; opening them does not require Python, a GPU, a CDN, or the stimulus images.

- `viewers/compare.html`: 33 measured state/run endpoints, including untrained, pretrained, DVD, label shuffle, SimCLR, transferred conditions, synthetic conditions and ResNet-50. Each panel displays its actual measured epoch. This is an inventory of the existing viewer's measurements, not a claim that all configured ICLR cohorts have been measured.
- `viewers/epochs.html`: 91 measured checkpoints, epochs 0–90, pretrained AlexNet `out_21`, seed 5959. This is a different run from the endpoint viewer's `out_04` baseline.

Choose a network and **Add panel** to compare any number of states. Camera rotation, zoom, point size/alpha, and scale are shared. **Compare all** adds every available state/sequence; rendering and export cost grow with panel count. Both pages accept multiple projection JSON files. The epoch page groups states by network, run, dataset hash and analysis protocol. Its timeline uses the actual epoch numbers, and only interpolates between supplied measurements. Epoch 0, 1, 90 and each sequence's first/last checkpoint are retained when available. Missing mandatory epochs are not fabricated. Image points always come from the nearest selected measured checkpoint, even during smooth interpolation.

GIF exports use the current camera, panels, scale and view mode at 50 fps. Duration is editable (default 8 seconds). SVG exports contain one PNG image per point cloud; axes, labels and ellipses remain editable vectors. SVG and GIF export restore the interactive controls and current epoch. PNG/SVG export resolution uses a common grid width, rather than the browser window's pixel dimensions.

## Paths in this checkout

```text
data/np/metadata.json                              Published stimulus manifest
data/np/set_<s>/N_<n>/w_<w>_b_<b>/<index>.png       Published images
data/responses/<network>/<run>/epoch_090.npz        Optional user-supplied responses
checkpoints/<state>/<run>/epoch_090.pth             Existing checkpoint contract
results/<network>/<run>/epoch_090/representation.json
results/viewers/bundled/endpoints.json              Shipped historical projections
results/viewers/bundled/pretrained_epochs.json      Shipped historical projections
configs/viewers.json                               Relative build input paths
viewers/assets/                                    Editable HTML/CSS/JS sources
viewers/compare.html
viewers/epochs.html
```

The historical bundles contain only 3D projections, covariance summaries, identifiers and source hashes. They do not contain checkpoints, images, full activations or machine-specific paths. Their original WB stimulus table is identified by its hash and labeled **Historical WB set123**. Generating new stimuli under `data/np` is not historical stimulus identity: new measurements are labeled by the actual published manifest hash. Cross-panel comparisons can contain different datasets or P6/P7 protocols; read the dataset/protocol labels in each panel before interpreting them as matched comparisons.

`data/np` currently contains a location guide. Populate it with the exact published images and `metadata.json` before collecting responses. Do not run inference on regenerated images while attaching the published manifest's identity. The response importer checks row paths and the manifest hash; it cannot verify whether a third-party model actually produced the submitted values from those images.

## Upload a network response from HTML

From the checkout root, after installing the existing project dependencies (`python -m pip install -e .`):

```bash
python -m iclr_repro.viewer template
python -m iclr_repro.viewer serve
```

Open `http://127.0.0.1:8765/viewers/compare.html` or `http://127.0.0.1:8765/viewers/epochs.html`, then choose **Load representations / responses** and select one or more `.npz` files. The local server converts them, adds their panels, and saves their projections under `results/<network>/<run>/epoch_<NNN>/representation.json`. The server binds only to localhost. It is not a public inference service. The HTML served by GitHub Pages or another static host accepts already-projected JSON; raw response conversion uses this local Python server or the CLI below.

**Template contract:** `data/responses/template/response_template.npz` contains:

| Key | Type | Meaning |
| --- | --- | --- |
| `responses` | numeric `[images, features]` | One flattened activation vector per nonblank image, at one fixed layer. The template has zero feature columns; replace it. At least three features and identifiable N1/P2 directions are required. |
| `image_paths` | Unicode `[images]` | Paths relative to `data/np`, aligned with response rows. Every nonblank image exactly once. Arbitrary row order is accepted and realigned by path. |
| `dataset_manifest_sha256` | Unicode scalar | SHA-256 of the exact `data/np/metadata.json` bytes. |
| `network_id` | Unicode scalar | Stable network/condition identifier; letters, digits, `_`, `-`. |
| `run` | Unicode scalar | Initialization/run identifier with the same allowed characters. |
| `label` | Unicode scalar | Human-readable model label. |
| `epoch` | integer scalar | Nonnegative measured epoch. Use the same network/run for an epoch sequence. |

The companion `image_order.csv` lists image hashes, N, w, b, set and folds. Responses must not contain NaN/Inf, object arrays or pickle payloads. Blanks (N=0) are excluded. All 495 nonblank cells and N=1–30 must be covered; the supplied metadata determines balanced folds. The template never silently fills synthetic activations.

An example for saving responses from any model is in `examples/save_network_responses.py`. The importer accepts arbitrary feature dimensions and architectures; no checkpoint architecture is assumed when importing responses. Keep one feature definition/layer per network sequence. Browser NPZ uploads are limited to 4 GiB compressed and 8 GiB expanded; very large responses require sufficient RAM. The CLI avoids the browser upload copy but also needs RAM for the response matrix. Raw input files supplied by users remain their inputs; the conversion process creates no additional raw response cache.

For command-line conversion:

```bash
python -m iclr_repro.viewer project data/responses/my_network/run_01/epoch_090.npz
# P7 explicitly uses the same conditioned7 function as ICLR geometry.py:
python -m iclr_repro.viewer project data/responses/my_network/run_01/epoch_090.npz --p-bins 7
```

P6 is the default to match the historical 3D viewers. P7 is selectable in the browser's raw response basis control. Conversion rejects mismatched dataset fingerprints, duplicated/missing paths, nonfinite responses and undefined N/P geometry. A malformed upload leaves the previously loaded panels available.

## Existing ICLR checkpoints

The extraction command follows `iclr_repro.paths.checkpoint` and the existing strict model loader, and checks each stimulus file's SHA-256. It performs no training:

```bash
python -m iclr_repro.viewer extract pretrained --run out_21 --epochs 0 1 2 3 90 --device cuda
```

Supply the epochs whose checkpoints actually exist; nothing is substituted for a missing checkpoint. The default ICLR training schedule does not itself promise all 91 checkpoints. For custom models, supply responses instead. Extracted full activations are transient in RAM; only `representation.json` is written. Existing analysis artifacts are preserved.

## Geometry and interpretation

The importer fits conditioned N rank 1 and P rank 2 on balanced folds B+C, then projects held-out fold A. Its orthonormal semantic frame is x=closest P direction to N, y=orthogonal P-plane direction, z=P-normal toward N. PCA signs are oriented deterministically. All arrows start at the actual held-out N=1 centroid. The large P-plane rectangle and a common isotropic range stay fixed across selected panels and all epochs; only its PC arrows change orientation.

Each exact-cell held-out Gaussian uses Ledoit–Wolf covariance in the same 3D space. The 36 display bins pool these moments, including between-cell mean dispersion. Historical endpoint bundles instead preserve their original direct-bin Ledoit–Wolf fits, explicitly labeled in every panel. No synthetic image activations are created. PA is the angle in the high-dimensional fitted N1/P2 geometry, independently displayed in a small angle diagram; screen-projected arrow angles depend on the camera.

The default **Activation normalized** scale preserves the original viewer's canonical scalar: equal mass for each N on 406 primary cells (w,b≥1, N≤29), using assigned A+B+C observations, and dividing total centered variance by feature count. This is display normalization; the N/P basis uses only B+C. **Projected RMS normalized** additionally offers a dimensionless projected-spread comparison; **Raw response units** preserves input scale. The same absolute range is used for every panel in the selected scale mode. The N=1 translation does not affect covariance or PA.

## Rebuild and share

Edit the asset sources, not the generated HTML. List any additional projection files in `configs/viewers.json` using paths inside this checkout, then run:

```bash
python -m iclr_repro.viewer build
```

Both HTML files embed their configured bundles. Ship them together for navigation; each can also open independently. Source bundles are explicitly exempted in `.gitignore`; private user responses and other result caches remain ignored. Do not place user responses into those bundled files unless you intend to share their 3D projections. To serve the complete documented package locally use `viewer serve`; a static deployment needs only the two generated HTML files (and this guide if desired).

Checks:

```bash
python -m unittest discover -s tests -p test_viewer.py -v
# Optional Node rendering/export checks; requires @napi-rs/canvas:
node tests/viewer_offline.cjs
```

The automated render test uses real Canvas pixels and the embedded GIF encoder outside a browser. It is not a claim of actual browser UI verification. Reports and preview images are written to ignored `results/viewers/qa/`.

The delivered pages were additionally opened in the Codex in-app browser through localhost. Point mode, adding a fifth comparison panel, synchronized camera input, GIF/SVG completion messages and epoch changes after export were checked. Raw NPZ conversion was exercised end to end through the localhost HTTP API with a synthetic fixture; this did not use the browser file chooser.
