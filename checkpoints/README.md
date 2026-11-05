# Network checkpoints

Training and analysis share `checkpoints/<state>/<run>/epoch_<NNN>.pth`. Expected dictionary keys: `backbone`, `state`, `run`, `seed`, `epoch`, `architecture`, `reference_implementation`. Architecture is `alexnet` or `resnet50`; AlexNet relu5 extraction requires every features.* convolution weight/bias. Do not substitute partial random weights. Training writes a relative-path-only `training_manifest.json` next to checkpoints. Contents are ignored by Git.
