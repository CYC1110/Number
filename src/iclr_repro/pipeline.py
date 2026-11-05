"""Explicit data -> checkpoint -> features -> estimators -> plot inputs."""
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from . import paths
from .geometry import registry, measure, conditioned7, cell_means
from .readouts import decoder_metrics, comparison_svm, context_generalization


def load_metadata(kind):
    root=paths.dataset(kind);manifest=root/"metadata.json"
    data=json.loads(manifest.read_text())
    table=pd.DataFrame(data["images"])
    if table.path.duplicated().any():raise ValueError("Duplicate stimulus file records could cross folds")
    if table.N.lt(0).any():raise ValueError("Negative numerosity in metadata")
    table=table.loc[table.N.gt(0)].reset_index(drop=True)
    for row in table.itertuples():
        filename=(root/row.path).resolve()
        if not filename.is_relative_to(root):raise ValueError("Stimulus path escapes dataset")
        if not filename.is_file():raise FileNotFoundError("Stimulus file missing")
        if not hasattr(row,"sha256") or hashlib.sha256(filename.read_bytes()).hexdigest()!=row.sha256:
            raise ValueError("Stimulus content hash missing or changed")
    table["b"]=table.N-table.w
    return table


def extract_features(state,run,epoch,table,kind="np",device="cpu",batch_size=32):
    from torchvision.models import alexnet,resnet50
    from torchvision import transforms
    checkpoint=paths.checkpoint(state,run,epoch)
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (saved.get("state"),saved.get("run"),int(saved.get("epoch",-1)))!=(state,run,epoch):
        raise ValueError("Checkpoint identity disagrees with requested state/run/epoch")
    seed=paths.run_seed(state,run)
    if saved.get("seed")!=seed:raise ValueError("Checkpoint initialization seed disagrees with configured mapping")
    architecture=saved["architecture"];weights=saved["backbone"]
    if architecture=="alexnet":
        network=alexnet(weights=None).features[:12]
        expected={k:v for k,v in weights.items() if k.startswith("features.")}
        expected={k.removeprefix("features."):v for k,v in expected.items()}
        network.load_state_dict(expected,strict=True)
    elif architecture=="resnet50":
        network=resnet50(weights=None);network.load_state_dict(weights,strict=True)
        network.fc=torch.nn.Identity()
    else:raise ValueError("Unsupported checkpoint architecture")
    network.to(device).eval()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(table),batch_size):
            images=[]
            for relative in table.path.iloc[start:start+batch_size]:
                with Image.open(paths.inside(paths.relative(paths.dataset(kind)),relative)) as image:
                    if image.size!=(227,227):raise ValueError("Evaluation images must be 227x227")
                    images.append(transform(image.convert("L").convert("RGB")))
            chunks.append(network(torch.stack(images).to(device)).flatten(1).cpu().numpy())
    x=np.concatenate(chunks)
    if not np.isfinite(x).all():raise ValueError("Nonfinite features")
    return x,{"checkpoint":paths.relative(checkpoint),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"seed":seed,
              "architecture":architecture,"feature_layer":"relu5" if architecture=="alexnet" else "avgpool",
              "feature_dimension":x.shape[1],"preprocessing":"227 grayscale repeated as RGB; ImageNet normalization"}


def analyze(state,run,epoch,device="cpu",basis_scope="full"):
    protocol=paths.protocol()
    seed=paths.run_seed(state,run)
    output=paths.result(state,run,epoch)
    if (output/"metrics.json").exists():raise FileExistsError("Analysis output exists; choose a fresh run/epoch")
    metadata=load_metadata("np")
    table,cells,balance=registry(metadata,protocol["split_seed"])
    if len(table)!=18000 or len(cells)!=495:raise ValueError("Canonical NP analysis requires 18,000 images / 495 cells")
    x,proof=extract_features(state,run,epoch,table,device=device)
    proof.update(dataset="data/np",dataset_manifest_sha256=hashlib.sha256((paths.dataset("np")/"metadata.json").read_bytes()).hexdigest())
    geometry,bases,displays=measure(x,table,cells)
    decoder_bases=bases
    if basis_scope=="train":
        # Optional stricter sensitivity; descriptive full-data default is preserved.
        m,c=cell_means(x,table,cells,np.flatnonzero(table.fold3_id.to_numpy()==0))
        fit=conditioned7(m,c,cells)
        decoder_bases={"U_N":fit["N"].basis[:,:1],"U_P":fit["P"].basis[:,:2]}
    reference=protocol["decoder"]
    options={"seed":protocol["split_seed"],"epochs":reference["epochs"],"lr":reference["learning_rate"]}
    decoders,models,predictions=decoder_metrics(x,table,decoder_bases,basis_scope=basis_scope,**options)
    comparison={target:comparison_svm(x,table,target,C=protocol["svm_C"],max_iter=protocol["svm_max_iter"]) for target in ["N","P"]}
    generalization={target:context_generalization(x,table,target,**options) for target in ["N","P"]}
    output.mkdir(parents=True,exist_ok=True)
    table.to_csv(output/"analysis_table.csv",index=False);balance.to_csv(output/"split_balance.csv",index=False)
    cells.to_csv(output/"cells.csv",index=False)
    np.savez_compressed(output/"bases.npz",**bases)
    np.savez_compressed(output/"test_predictions.npz",**predictions)
    for held,display in enumerate(displays):np.savez_compressed(output/f"geometry_fold_{held}.npz",**display)
    for target,model in models.items():torch.save(model.state_dict(),output/f"decoder_{target}.pth")
    metrics={"state":state,"run":run,"epoch":epoch,"seed":seed,"expected_networks_in_state":len(paths.runs(state)),"purpose":protocol["purpose"],
             "geometry":geometry,"decoders":decoders,"comparison":comparison,"generalization":generalization,
             "basis_scope":basis_scope,"source":proof,
             "split":{"seed":protocol["split_seed"],"decoder_train":0,"decoder_validation":1,"decoder_test":2,
                      "unused_remainders":"fold3_id=-1; excluded from readouts and BC; retained in descriptive full-data PCA"}}
    paths.write_json(output/"metrics.json",metrics)
    return metrics


def analyze_units(state,run,epoch,device="cpu"):
    from .estimators.selectivity import SelectivityConfig,compute_selectivity
    protocol=paths.protocol();seed=paths.run_seed(state,run);table=load_metadata("numerosity")
    table,cells,_=registry(table,protocol["split_seed"])
    x,proof=extract_features(state,run,epoch,table,kind="numerosity",device=device)
    proof.update(dataset="data/numerosity",dataset_manifest_sha256=hashlib.sha256((paths.dataset("numerosity")/"metadata.json").read_bytes()).hexdigest())
    train=np.flatnonzero(table.fold3_id.to_numpy()==0)
    counts=table.iloc[train].groupby(["N","set_id"]).size()
    if counts.nunique()!=1:raise ValueError("Preserved ANOVA requires a balanced complete factorial design")
    unit=compute_selectivity(x[train],table.N.to_numpy()[train],table.set_id.to_numpy()[train],SelectivityConfig())
    chosen=unit["is_selective"];unselected=np.flatnonzero(~chosen);selected=np.flatnonzero(chosen)
    output=paths.result(state,run,epoch)/"single_unit"
    if (output/"metrics.json").exists():raise FileExistsError("Single-unit output already exists")
    output.mkdir(parents=True,exist_ok=True)
    # Match the number of nonselective units and average five deterministic draws.
    rng=np.random.default_rng(protocol["split_seed"]);n=min(len(selected),len(unselected),256)
    scores=[]
    if n:
        for draw in range(5):
            si=rng.choice(selected,n,replace=False);ni=rng.choice(unselected,n,replace=False)
            scores.append({"draw":draw,"selective":comparison_svm(x[:,si],table,"N",restrict_gap=False),
                           "nonselective":comparison_svm(x[:,ni],table,"N",restrict_gap=False)})
    # Averaged tuning (all stimuli) is descriptive, not a test-set score.
    levels=sorted(table.N.unique());tuning=np.array([x[table.N.to_numpy()==n].mean(0) for n in levels])
    np.savez_compressed(output/"selectivity.npz",**unit,numerosities=levels,tuning=tuning)
    result={"state":state,"run":run,"epoch":epoch,"seed":seed,"expected_networks_in_state":len(paths.runs(state)),"source":proof,"alpha":.01,"interaction_alpha":.01,
            "multiplicity_correction":"none; original definition retained","selection_scope":"decoder train fold only",
            "selected_fraction":float(chosen.mean()),"comparison_units":n,"svm_draws":scores,
            "selection_note":"Reference train-only selection differs from historical independent shared-cache selection"}
    paths.write_json(output/"metrics.json",result)
    return result
