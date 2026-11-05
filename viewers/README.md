# Two offline viewers

Open `compare.html` for measured network endpoints or `epochs.html` for the 91 pretrained checkpoints. Both support multiple synchronized panels and uploaded projection JSON files. Raw NPZ uploads use the optional local server, `python -m iclr_repro.viewer serve`.

See `../docs/VIEWERS.md` for data templates, relative paths, projection methods, commands and sharing. Edit `assets/` and rebuild with `python -m iclr_repro.viewer build`. Generated HTML embeds all required data and code.
