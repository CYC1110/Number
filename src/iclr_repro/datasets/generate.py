"""Clean dataset CLIs: plans are default; generation must be requested explicitly."""
from dataclasses import fields
import csv
import hashlib
import json
import shutil
import numpy as np
from PIL import Image
from .. import paths


def np_plan():
    settings=json.loads(paths.inside("configs","np_dataset.json").read_text())
    return {"output":paths.relative(paths.dataset("np")),"seed":settings["config"]["seed"],
            "images_with_blanks":18600,"analysis_images":18000,"nonblank_cells":495,
            "regimes":["constant dot radius","constant color-specific total area","pentagonal spatial extent"],
            "set2_total_area":8000,"set2_min_dot_area":113,"status":"plan; no images generated"}


def generate_np():
    from .dot_core import StimuliConfig, _rng, _valid_combos, _combo_counts, _assign_colors, _draw_dots
    from .dot_regimes import _generate_set1_positions_tracked, _generate_set3_positions_tracked, _make_set2_positions
    settings=json.loads(paths.inside("configs","np_dataset.json").read_text());raw=settings["config"]
    kwargs={f.name:raw[f.name] for f in fields(StimuliConfig) if f.name in raw}
    cfg=StimuliConfig(**kwargs);rng=_rng(cfg.seed);out=paths.dataset("np")
    if (out/"metadata.json").exists() or any(out.glob("set_*")):
        raise FileExistsError("NP images already exist; generation never overwrites a dataset")
    plans=[]
    # Freeze all allocation draws before drawing images, as in the original.
    for regime in settings["sets"]:
        for n in cfg.number_sets:
            combinations=_valid_combos(n,cfg.w_values,cfg.b_values)
            counts=_combo_counts(rng,cfg.repeats,len(combinations),cfg.per_combo_repeats)
            plans.extend((regime,n,w,b,int(count)) for (w,b),count in zip(combinations,counts))
    records=[];out.mkdir(parents=True,exist_ok=True)
    for regime,n,w,b,count in plans:
        for index in range(count):
            common={"placement_max_attempts_total":raw["placement_max_attempts_total"],
                    "placement_per_item_attempts":raw["placement_per_item_attempts"]}
            if regime==2:
                colors=_assign_colors(n,w,rng)
                positions,audit=_make_set2_positions(n,colors,rng,cfg,min_area_per_dot=raw["set2_min_area_per_dot"],
                    set2_area_attempts=raw["set2_area_attempts"],set2_distance_attempts=raw["set2_distance_attempts"],
                    enforce_distance_match=raw["set2_distance_match"],**common)
            else:
                generator=_generate_set1_positions_tracked if regime==1 else _generate_set3_positions_tracked
                positions,audit=generator(n,rng,cfg,layout_attempts=raw["layout_attempts"],**common)
                colors=_assign_colors(n,w,rng)
            if not audit["generation_success"]:raise RuntimeError("Dot placement failed; do not silently retain incomplete dots")
            pixels=_draw_dots(cfg.H,cfg.W,positions,colors,cfg)
            relative=f"set_{regime}/N_{n}/w_{w}_b_{b}/{index:05d}.png"
            target=paths.inside("data","np",relative);target.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(pixels).save(target)
            records.append({"path":relative,"N":n,"w":w,"b":b,"set_id":regime,"index":index,
                            "sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"placement":audit})
    paths.write_json(out/"metadata.json",{"config":raw,"images":records,"sets":settings["sets"]})


def synthetic_plan(kind, count=None):
    count=paths.protocol()["training_images"] if count is None else int(count)
    if count<=0:raise ValueError("Image count must be positive")
    return {"condition":kind,"count":count,"seed":paths.protocol()["synthetic_dataset_seed"],
            "resolution":128 if kind=="statistical" else 256,
            "output":paths.relative(paths.dataset(kind)),
            "source":"official local WMM PNGs" if kind=="statistical" else "local shader source manifest" if kind=="shaders" else "procedural",
            "status":"plan; no images generated"}


def generate_synthetic(kind,count=None):
    plan=synthetic_plan(kind,count);out=paths.dataset(kind)
    if (out/"metadata.json").exists() or any(out.glob("*.png")):
        raise FileExistsError("Synthetic images already exist")
    seed=plan["seed"];out.mkdir(parents=True,exist_ok=True)
    if kind=="noise":
        from .noise import NoiseSource
        source=NoiseSource("noise",256,seed)
    elif kind=="grating":
        from .grating import grating
        source=lambda i:grating(i,256,seed)
    elif kind=="shaders":
        from .shader import ShaderRenderer
        manifest=paths.inside("data","assets","shaders","manifest.json")
        if not manifest.exists():raise FileNotFoundError("Supply local shader manifest and source files first")
        # The original renderer resolves relative source paths against manifest.parent.
        data=json.loads(manifest.read_text())
        for row in data.get("programs",[]):
            asset=(manifest.parent/row["assembled_path"]).resolve()
            if not asset.is_relative_to(paths.ROOT):raise ValueError("Shader source escapes checkout")
        source=ShaderRenderer(manifest,256,seed)
    elif kind=="statistical":
        # Official Spectrum + Color + WMM is selected from supplied data, not
        # approximated by random noise or an unverified texture generator.
        originals=sorted(paths.inside("data","assets","wmm_official").rglob("*.png"))
        if len(originals)<plan["count"]:raise ValueError("Not enough official local WMM images")
        ids=np.sort(np.random.default_rng(seed).choice(len(originals),plan["count"],replace=False))
        originals=[originals[i] for i in ids]
        source=None
    else:raise ValueError("Unknown synthetic condition")
    records=[]
    try:
        for i in range(plan["count"]):
            filename=f"{i:08d}.png";target=paths.inside(paths.relative(out),filename)
            if source is None:
                original=originals[i].resolve()
                if not original.is_relative_to(paths.ROOT):raise ValueError("WMM source escapes checkout")
                with Image.open(original) as image:
                    if image.size!=(128,128) or image.mode!="RGB":raise ValueError("Official WMM reference source must be RGB 128x128")
                shutil.copyfile(original,target)
            else:Image.fromarray(source(i)).save(target)
            records.append({"path":filename,"sample_id":i,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    finally:
        if source is not None and hasattr(source,"close"):source.close()
    paths.write_json(out/"metadata.json",{**plan,"status":"generated","images":records})


def numerosity_plan():
    return {"output":"data/numerosity","seed":0,"images":2400,"number_levels":[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30],
            "sets":3,"repeats_per_cell":50,"status":"plan; no images generated"}


def generate_numerosity():
    from .numerosity_core import StimuliConfig,generate_stimuli
    out=paths.dataset("numerosity")
    if (out/"metadata.json").exists() or any(out.glob("*.png")):raise FileExistsError("Numerosity dataset already exists")
    generated=generate_stimuli(StimuliConfig(seed=0,auto_cache=False,cache_path=None))
    pixels,numbers,regimes=[generated[k] for k in ["images","numeros","set_id"]]
    records=[];out.mkdir(parents=True,exist_ok=True)
    for i,(image,n,regime) in enumerate(zip(pixels,numbers,regimes)):
        filename=f"{i:05d}.png";target=out/filename
        Image.fromarray(image).save(target)
        records.append({"path":filename,"N":int(n),"w":int(n),"b":0,"set_id":int(regime),
                        "sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
    paths.write_json(out/"metadata.json",{"config":numerosity_plan(),"images":records})
