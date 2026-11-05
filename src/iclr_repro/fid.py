"""FID on local frozen Clean-FID features; no model downloads."""
import json
import numpy as np
from scipy.linalg import sqrtm
from . import paths


def frechet_distance(first,second):
    first,second=np.asarray(first,float),np.asarray(second,float)
    if first.ndim!=2 or second.ndim!=2 or first.shape[1]!=second.shape[1] or min(len(first),len(second))<2:
        raise ValueError("Need at least two finite samples from each feature distribution")
    if not np.isfinite(first).all() or not np.isfinite(second).all():raise ValueError("Nonfinite FID features")
    mu_a,mu_b=first.mean(0),second.mean(0)
    a,b=np.cov(first,rowvar=False),np.cov(second,rowvar=False)
    root=sqrtm(a@b)
    if np.iscomplexobj(root):
        if np.max(np.abs(root.imag))>1e-6:raise ValueError("Complex covariance square root; no epsilon repair")
        root=root.real
    value=float(np.sum((mu_a-mu_b)**2)+np.trace(a+b-2*root))
    if value < -1e-6:raise ValueError("Negative FID beyond rounding tolerance")
    return max(value,0.)


def compute_local_fid():
    root=paths.inside("data","fid_features")
    features={};metadata={}
    for state in ["imagenet","noise","grating","statistical","shaders"]:
        features[state]=np.load(root/f"{state}.npy",allow_pickle=False)
        metadata[state]=json.loads((root/f"{state}.json").read_text())
        if features[state].shape!=(50000,2048):raise ValueError("Clean-FID contract requires 50,000 x 2,048 local features")
        if metadata[state].get("extractor")!="clean-fid-0.1.35-clean":raise ValueError("Unverified feature protocol")
        if metadata[state].get("unique_image_count")!=50000:raise ValueError("FID sample uniqueness not attested")
    hashes={d.get("extractor_weight_sha256") for d in metadata.values()}
    if len(hashes)!=1 or not next(iter(hashes)):raise ValueError("FID extractor weights must match across conditions")
    results={s:frechet_distance(features["imagenet"],features[s]) for s in ["noise","grating","statistical","shaders"]}
    paths.write_json(paths.inside("results","fid.json"),{"fid":results,"n_images":50000,"extractor":"clean-fid-0.1.35-clean","weights_sha256":next(iter(hashes))})
    return results
