"""Save activations produced by YOUR model using the dataset-specific template.

First run: python -m iclr_repro.viewer template
Import save_responses from your inference script; do not change the image order
without passing the corresponding image_paths to it.
"""
import numpy as np
from iclr_repro import paths


def save_responses(responses, *, network_id, label, run='run_01', epoch=90, image_paths=None):
    template=paths.inside('data/responses/template/response_template.npz')
    with np.load(template, allow_pickle=False) as t:
        order=t['image_paths'].copy() if image_paths is None else np.asarray(image_paths,dtype=str)
        fingerprint=t['dataset_manifest_sha256'].copy()
    x=np.asarray(responses,dtype=np.float32)
    if x.ndim!=2 or len(x)!=len(order) or x.shape[1]<3 or not np.isfinite(x).all():
        raise ValueError('Expected finite [images,features>=3] activations aligned to image_paths')
    output=paths.inside('data/responses',paths.identifier(network_id),paths.identifier(run),f'epoch_{int(epoch):03d}.npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,responses=x,image_paths=order,dataset_manifest_sha256=fingerprint,
                        network_id=network_id,label=label,run=run,epoch=int(epoch))
    return output

# In your own inference script:
# with np.load(paths.inside('data/responses/template/response_template.npz')) as t:
#     image_paths = t['image_paths'].copy()
# responses = your_model_responses_in_this_order(image_paths)
# save_responses(responses, network_id='my_network', label='My network', epoch=90)
# Then select the resulting NPZ in the local HTML viewer, or run viewer project.
