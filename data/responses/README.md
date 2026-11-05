# User-supplied network responses

Create the image-order template with `python -m iclr_repro.viewer template` after populating `data/np/metadata.json` with the published dataset manifest. Save responses under `<network>/<run>/epoch_<NNN>.npz`. Format and validation are documented in `docs/VIEWERS.md`. Raw response files stay ignored; conversion saves only 3D geometry under `results/<network>/<run>/epoch_<NNN>/representation.json`.
