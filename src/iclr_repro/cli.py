"""The CLI defaults to readable plans; heavy actions require --execute."""
import argparse
import json
from . import paths


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("paths",help="Show central relative data/checkpoint contracts")
    p=sub.add_parser("generate-np");p.add_argument("--execute",action="store_true")
    p=sub.add_parser("generate-numerosity");p.add_argument("--execute",action="store_true")
    p=sub.add_parser("generate-synthetic");p.add_argument("kind",choices=["noise","grating","statistical","shaders"])
    p.add_argument("--count",type=int);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("train");p.add_argument("state");p.add_argument("--run",default="out_04")
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("analyze");p.add_argument("state");p.add_argument("--run",default="out_04");p.add_argument("--epoch",type=int)
    p.add_argument("--device",default="cpu");p.add_argument("--basis-scope",choices=["full","train"],default="full")
    p.add_argument("--single-unit",action="store_true");p.add_argument("--execute",action="store_true")
    p=sub.add_parser("plot");p.add_argument("figure",choices=["F1","F2","F3","F4","F5","F6","S1","S2","S3","S4","S5","S6","S7"])
    p.add_argument("--execute",action="store_true")
    p=sub.add_parser("audit",help="Static privacy/path/coverage checks");p.add_argument("--write-report",action="store_true")
    p=sub.add_parser("fid",help="FID from supplied local Clean-FID feature arrays");p.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    if args.command=="paths":
        result={"data":"data/<dataset>","checkpoint":"checkpoints/<state>/<run>/epoch_<NNN>.pth",
                "analysis":"results/<state>/<run>/epoch_<NNN>","figures":"figures/<figure>.svg","networks_by_state":paths.state_counts()}
    elif args.command in ["generate-np","generate-numerosity","generate-synthetic"]:
        from .datasets.generate import np_plan,generate_np,synthetic_plan,generate_synthetic,numerosity_plan,generate_numerosity
        if args.command=="generate-np":
            result=np_plan()
            if args.execute:generate_np();result={"status":"generated"}
        elif args.command=="generate-numerosity":
            result=numerosity_plan()
            if args.execute:generate_numerosity();result={"status":"generated"}
        else:
            result=synthetic_plan(args.kind,args.count)
            if args.execute:generate_synthetic(args.kind,args.count);result={"status":"generated"}
    elif args.command=="train":
        from .training import training_plan,train
        result=training_plan(args.state,args.run)
        if args.execute:train(args.state,args.run,args.device,args.batch_size);result={"status":"trained"}
    elif args.command=="analyze":
        protocol=paths.protocol()
        try:seed=paths.run_seed(args.state,args.run)
        except ValueError as error:parser.error(str(error))
        epoch=args.epoch if args.epoch is not None else protocol["endpoint_epochs"].get(args.state,protocol["default_endpoint_epoch"])
        result={"state":args.state,"run":args.run,"seed":seed,"expected_networks_in_state":len(paths.runs(args.state)),"epoch":epoch,"checkpoint":paths.relative(paths.checkpoint(args.state,args.run,epoch)),
                "output":paths.relative(paths.result(args.state,args.run,epoch)),"basis_scope":args.basis_scope,"status":"plan; no analysis run"}
        if args.execute:
            from .pipeline import analyze,analyze_units
            result=analyze_units(args.state,args.run,epoch,args.device) if args.single_unit else analyze(args.state,args.run,epoch,args.device,args.basis_scope)
    elif args.command=="plot":
        result={"figure":args.figure,"status":"plan; requires completed inputs described in docs/FIGURES.md"}
        if args.execute:
            from .figures import render
            result=render(args.figure)
    elif args.command=="fid":
        result={"status":"plan; no FID computed","features":"data/fid_features/<condition>.npy + .json","shape":[50000,2048]}
        if args.execute:
            from .fid import compute_local_fid
            result=compute_local_fid()
    else:
        from .audit import audit
        result=audit()
        if args.write_report:paths.write_json(paths.inside("docs","STATIC_AUDIT.json"),result)
    print(json.dumps(result,indent=2,default=str))
