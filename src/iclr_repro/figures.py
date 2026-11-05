"""Method figures from condition-specific cohorts, with explicit missing-input failures."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import t as student_t
from PIL import Image
from . import paths
from .readouts import paired_statistics,state_correlation

BASE=["label_shuffled","untrained","pretrained","dvd","simclr","rotnet","dino"]
SYNTHETIC=["noise","grating","statistical","shaders"]


def endpoint(state):
    p=paths.protocol();return p["endpoint_epochs"].get(state,p["default_endpoint_epoch"])


def collect(states,epoch=None,units=False):
    records={};dataset_hashes=set();basis_scopes=set()
    for state in states:
        rows=[]
        for run in paths.runs(state):
            root=paths.result(state,run,endpoint(state) if epoch is None else epoch)
            if units:root=root/"single_unit"
            file=root/"metrics.json"
            if not file.exists():raise FileNotFoundError(f"Required configured-cohort input missing: {paths.relative(file)}")
            row=json.loads(file.read_text())
            expected_epoch=endpoint(state) if epoch is None else epoch
            if (row.get("state"),row.get("run"),row.get("epoch"))!=(state,run,expected_epoch):
                raise ValueError("Metrics identity disagrees with state/run/epoch")
            expected_seed=paths.run_seed(state,run)
            if row.get("seed")!=expected_seed or row["source"].get("seed")!=expected_seed:
                raise ValueError("Metrics/checkpoint seed disagrees with configured run mapping")
            if row.get("expected_networks_in_state")!=len(paths.runs(state)):
                raise ValueError("Metrics cohort count disagrees with current configuration")
            dataset_hashes.add(row["source"]["dataset_manifest_sha256"])
            if not units:basis_scopes.add(row["basis_scope"])
            rows.append(row)
        if len(rows)!=len(paths.runs(state)):raise ValueError("Every plotted state needs its entire configured cohort")
        records[state]=rows
    if len(dataset_hashes)!=1:raise ValueError("Figure inputs use different stimulus manifests")
    if len(basis_scopes)>1:raise ValueError("Do not mix full-data and train-only ablation analyses")
    return records


def value(row,key):
    current=row
    for part in key.split("."):current=current[part]
    if current is None or not np.isfinite(current):raise ValueError(f"Undefined canonical value: {key}")
    return float(current)


def samples(records,state,key):return np.array([value(row,key) for row in records[state]])


def network_summary(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError("Network summary needs at least two finite network observations")
    return {"n":len(values),"mean":float(values.mean()),"sd":float(values.std(ddof=1)),
            "sem":float(values.std(ddof=1)/np.sqrt(len(values)))}


def seed_values(records,state,key):
    rows=records[state];mapped={row["seed"]:value(row,key) for row in rows}
    if len(mapped)!=len(rows):raise ValueError("Duplicate initialization seed in cohort")
    return mapped


def map_panel(ax,records,states):
    for state in states:
        x=samples(records,state,"geometry.principal_angle_deg")
        y=samples(records,state,"geometry.all_pair_bc")
        ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt="o",label=state)
    ax.set(xlabel="Principal angle (degrees)",ylabel="All-pair BC")
    ax.legend(fontsize=7)


def bar_panel(ax,records,states,key,label):
    means=[];errors=[]
    for state in states:
        summary=network_summary(samples(records,state,key));means.append(summary["mean"]);errors.append(summary["sem"])
    ax.bar(np.arange(len(states)),means,yerr=errors,capsize=3)
    ax.set_xticks(np.arange(len(states)),states,rotation=35,ha="right");ax.set_ylabel(label)


def correlation_panel(ax,records,x_key,y_key):
    states=list(records)
    x=np.array([samples(records,s,x_key).mean() for s in states])
    y=np.array([samples(records,s,y_key).mean() for s in states])
    stats=state_correlation(x,y);ax.scatter(x,y)
    for state,a,b in zip(states,x,y):ax.annotate(state,(a,b),fontsize=6)
    coefficients=np.polyfit(x,y,1);grid=np.linspace(x.min(),x.max(),100)
    residual=y-np.polyval(coefficients,x);df=len(x)-2;sxx=np.sum((x-x.mean())**2)
    if sxx<=0:raise ValueError("Constant predictor cannot define an OLS fit")
    se=np.sqrt(np.sum(residual**2)/df*(1/len(x)+(grid-x.mean())**2/sxx))
    fit=np.polyval(coefficients,grid);band=student_t.ppf(.975,df)*se
    ax.plot(grid,fit,"--");ax.fill_between(grid,fit-band,fit+band,alpha=.15)
    ax.set(xlabel=x_key,ylabel=y_key,title=f"r={stats['pearson_r']:.2f}, p={stats['p_two_sided']:.3g}")
    return {"x":x_key,"y":y_key,"states":states,**stats,"confidence_band":"95% OLS mean CI"}


def geometry_picture(ax,state="pretrained"):
    root=paths.result(state,"out_04",endpoint(state))
    with np.load(root/"geometry_fold_0.npz",allow_pickle=False) as data:
        q,mu,cov,un,up=[data[k] for k in ["Q","means","covariances","U_N","U_P"]]
    n=q.T@un[:,0];p=q.T@up[:,0];p=p-n*np.dot(n,p)
    if np.linalg.norm(p)<1e-12:raise ValueError("N-PC1 / orthogonal P-PC1 display undefined")
    axes=np.column_stack([n,p/np.linalg.norm(p)])
    centers=mu@axes;cov2=np.einsum("ai,nab,bj->nij",axes,cov,axes)
    for m,c in zip(centers,cov2):
        eig,rotation=np.linalg.eigh(c)
        if eig.min()<=0:raise ValueError("Display covariance undefined")
        direction=rotation[:,-1];angle=np.degrees(np.arctan2(direction[1],direction[0]))
        ax.add_patch(Ellipse(m,2*np.sqrt(eig[-1]),2*np.sqrt(eig[0]),angle=angle,fill=False,alpha=.15,lw=.4))
    ax.scatter(centers[:,0],centers[:,1],s=4);ax.autoscale_view()
    ax.set(xlabel="N-PC1",ylabel="orth(P-PC1 | N-PC1)",title="out_04; held-out A, basis B+C; 1-SD cells")


def render(figure):
    manifest={"figure":figure,"purpose":"method_reference","configured_networks_by_state":paths.state_counts(),
              "artwork_note":"New method layout; no claim of original Illustrator artwork reproduction"}
    if figure=="F1":
        fig,ax=plt.subplots(figsize=(10,3));ax.axis("off")
        labels=["Visual diet","Train frozen backbone","NP stimuli -> activations","PCA N1 / P2 + held-out BC","Readouts / transfer / ablation"]
        for i,label in enumerate(labels):
            ax.text(.03+i*.195,.5,label,ha="left",va="center",wrap=True,fontsize=8,
                    bbox={"boxstyle":"round","facecolor":"white","edgecolor":"black"},transform=ax.transAxes)
            if i<4:ax.annotate("",xy=(.22+i*.195,.5),xytext=(.18+i*.195,.5),arrowprops={"arrowstyle":"->"},xycoords="axes fraction")
    elif figure=="F2":
        states=["untrained","pretrained"];units=collect(states,units=True);np_records=collect(states)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        bar_panel(axes[0],units,states,"selected_fraction","Numerosity selective fraction (SEM)")
        # Actual independent-image SVM results, not copied manuscript values.
        reduced={}
        for state,rows in units.items():
            reduced[state]=[{"svm":float(np.mean([d["selective"]["accuracy"] for d in r["svm_draws"]]))} for r in rows]
        bar_panel(axes[1],reduced,states,"svm","Selective-unit comparison SVM")
        bar_panel(axes[2],np_records,states,"comparison.P.incongruent_accuracy","NP incongruent P comparison")
        manifest["external_panel"]="Human developmental data/inset not provided or re-created"
    elif figure=="F3":
        states=["untrained","pretrained"];records=collect(states);fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,key,label in zip(axes[:2],["geometry.principal_angle_deg","geometry.all_pair_bc"],["Principal angle (degrees; SEM)","All-pair BC (SEM)"]):
            bar_panel(ax,records,states,key,label)
            manifest.setdefault("paired_tests",{})[key]=paired_statistics(seed_values(records,states[0],key),seed_values(records,states[1],key))
        geometry_picture(axes[2])
    elif figure in ["F4","F5"]:
        states=BASE if figure=="F4" else ["untrained"]+SYNTHETIC
        records=collect(states);fig,ax=plt.subplots(figsize=(8,6));map_panel(ax,records,states)
        manifest["error_bars"]="Sample SD across each complete configured cohort"
    elif figure in ["F6","S2"]:
        records=collect(BASE+SYNTHETIC)
        specifications=[("geometry.all_pair_bc","decoders.NP.accuracy"),
                        ("geometry.all_pair_bc","comparison.P.incongruent_accuracy"),
                        ("geometry.principal_angle_deg","generalization.N.accuracy"),
                        ("geometry.principal_angle_deg","decoders.N.cross_quantity_damage")]
        if figure=="S2":
            specifications=[(x,y) for x in ["geometry.all_pair_bc","geometry.principal_angle_deg"]
                            for y in ["decoders.NP.accuracy","comparison.N.incongruent_accuracy","comparison.P.incongruent_accuracy",
                                      "generalization.N.accuracy","generalization.P.accuracy","decoders.N.cross_quantity_damage","decoders.P.cross_quantity_damage"]]
        columns=2;fig,axes=plt.subplots(int(np.ceil(len(specifications)/columns)),columns,figsize=(12,4*np.ceil(len(specifications)/columns)))
        manifest["correlations"]=[correlation_panel(ax,records,x,y) for ax,(x,y) in zip(np.ravel(axes),specifications)]
    elif figure=="S1":
        metadata=json.loads((paths.dataset("np")/"metadata.json").read_text())["images"]
        fig,axes=plt.subplots(3,5,figsize=(10,6))
        for row,regime in enumerate([1,2,3]):
            for col,w in enumerate([0,2,5,8,10]):
                item=next(x for x in metadata if (x["set_id"],x["N"],x["w"])==(regime,10,w))
                with Image.open(paths.inside("data","np",item["path"])) as image:axes[row,col].imshow(image,cmap="gray",vmin=0,vmax=255)
                axes[row,col].set_title(f"set {regime}, N=10, W={w}",fontsize=7);axes[row,col].axis("off")
    elif figure=="S3":
        records=collect(BASE+SYNTHETIC);fig,axes=plt.subplots(1,2,figsize=(10,4))
        for target,ax in zip(["N","P"],axes):
            for state,rows in records.items():
                values=[np.array(r["geometry"][target+"_cumulative_ev"],float) for r in rows]
                count=min(map(len,values));array=np.stack([v[:count] for v in values]);grid=np.arange(1,count+1)
                ax.plot(grid,array.mean(0),label=state);ax.fill_between(grid,array.mean(0)-array.std(0,ddof=1),array.mean(0)+array.std(0,ddof=1),alpha=.08)
            ax.axhline(.9,ls=":",color="black");ax.set(title=target,xlabel="Retained components",ylabel="Cumulative explained variance")
            ax.legend(fontsize=6)
        manifest["threshold_note"]="90% is descriptive; fixed N1/P2 need not explain >90% in every state"
    elif figure=="S4":
        states=["label_shuffled","correct_05","correct_10","correct_20","correct_50","pretrained"]
        records=collect(states);fig,axes=plt.subplots(1,2,figsize=(12,5));map_panel(axes[1],records,states)
        for state in states:
            runs=[]
            for run in paths.runs(state):
                row=json.loads((paths.checkpoint(state,run,0).parent/"training_manifest.json").read_text())
                if (row.get("state"),row.get("run"),row.get("seed"),row.get("expected_networks_in_state"))!=(state,run,paths.run_seed(state,run),len(paths.runs(state))):
                    raise ValueError("Validation history identity/seed/cohort mismatch")
                history=row["validation_top1"]
                if len(history)!=endpoint(state) or not np.isfinite(history).all():
                    raise ValueError("Validation history must cover every endpoint epoch with finite values")
                runs.append(history)
            lengths={len(v) for v in runs}
            if len(lengths)!=1:raise ValueError("Validation trajectories need a common epoch schedule within the configured cohort")
            values=np.array(runs);epochs=np.arange(1,values.shape[1]+1)
            axes[0].plot(epochs,values.mean(0),label=state);axes[0].fill_between(epochs,values.mean(0)-values.std(0,ddof=1),values.mean(0)+values.std(0,ddof=1),alpha=.1)
        axes[0].set(xlabel="Epoch",ylabel="ImageNet validation top-1");axes[0].legend(fontsize=7)
    elif figure=="S5":
        fig,ax=plt.subplots(figsize=(10,7))
        for state in BASE+SYNTHETIC+["resnet50_pretrained"]:
            epochs=[e for e in paths.protocol()["checkpoint_epochs"] if e<=endpoint(state)]
            x=[];y=[]
            for epoch in epochs:
                record=collect([state],epoch=epoch)
                x.append(samples(record,state,"geometry.principal_angle_deg").mean());y.append(samples(record,state,"geometry.all_pair_bc").mean())
            ax.plot(x,y,"o-",label=state)
            for a,b,epoch in zip(x,y,epochs):ax.annotate(str(epoch),(a,b),fontsize=5)
        ax.set(xlabel="Principal angle (degrees)",ylabel="All-pair BC");ax.legend(fontsize=7)
        manifest["cohort"]="Same condition-specific configured runs at every milestone; no interpolation or finite-only mean"
    elif figure=="S6":
        results=json.loads(paths.inside("results","fid.json").read_text());fig,axes=plt.subplots(2,4,figsize=(10,5))
        for i,state in enumerate(SYNTHETIC):
            axes[0,i].bar([0],[results["fid"][state]]);axes[0,i].set_title(state);axes[0,i].set_xticks([])
            images=sorted(paths.dataset(state).glob("*.png"))
            if not images:raise FileNotFoundError("Actual example images are required for S6")
            with Image.open(images[0]) as image:axes[1,i].imshow(image)
            axes[1,i].axis("off")
        manifest["fid_protocol"]=results
    elif figure=="S7":
        return viewer()
    else:raise ValueError("Unknown figure")
    if "records" in locals():
        manifest["observed_networks_by_state"]={state:len(rows) for state,rows in records.items()}
    if figure=="F2":manifest["observed_networks_by_state"]={state:len(rows) for state,rows in units.items()}
    if figure=="S5":manifest["observed_networks_by_state"]={state:len(paths.runs(state)) for state in BASE+SYNTHETIC+["resnet50_pretrained"]}
    fig.suptitle(f"{figure}: reference methods",fontsize=11)
    fig.tight_layout();output=paths.inside("figures",figure+".svg")
    plt.rcParams["svg.fonttype"]="none"
    fig.savefig(output,metadata={"Creator":"quantity-geometry-reference"});plt.close(fig)
    paths.write_json(output.with_suffix(".json"),manifest)
    return {"output":paths.relative(output),"manifest":paths.relative(output.with_suffix(".json"))}


def viewer():
    """Self-contained offline 3D centroid/ellipsoid viewer; no CDN dependencies."""
    root=paths.result("pretrained","out_04",90)
    with np.load(root/"geometry_fold_0.npz",allow_pickle=False) as data:
        q,un,up=data["Q"],data["U_N"],data["U_P"]
        payload={"means":data["means"].tolist(),"covariances":data["covariances"].tolist(),
                 "axes":[(q.T@un[:,0]).tolist(),(q.T@up[:,0]).tolist(),(q.T@up[:,1]).tolist()]}
    html='''<!doctype html><meta charset="utf-8"><title>Quantity geometry</title>
<style>body{font-family:sans-serif;margin:2em}canvas{border:1px solid #ccc;touch-action:none}</style>
<h1>Held-out quantity geometry</h1><p>Pretrained / out_04; A display, B+C basis. Drag to rotate. Ellipses show projected one-SD cell covariances.</p>
<canvas id="view" width="900" height="700"></canvas><script>
const data=PAYLOAD,c=document.getElementById('view'),ctx=c.getContext('2d');let yaw=.4,pitch=.3,last=null;
function dot(a,b){return a.reduce((s,v,i)=>s+v*b[i],0)}
function render(){const ex=[Math.cos(yaw),0,-Math.sin(yaw)],ey=[Math.sin(yaw)*Math.sin(pitch),Math.cos(pitch),Math.cos(yaw)*Math.sin(pitch)];
const xy=data.means.map(m=>[dot(m,ex),dot(m,ey)]),extent=Math.max(1,...xy.flat().map(Math.abs)),scale=250/extent;
ctx.clearRect(0,0,900,700);for(let i=0;i<xy.length;i++){const p=xy[i],cov=data.covariances[i];
const product=v=>cov.map(r=>dot(r,v)),a=dot(ex,product(ex)),b=dot(ex,product(ey)),d=dot(ey,product(ey)),gap=Math.hypot(a-d,2*b);
const l1=(a+d+gap)/2,l2=(a+d-gap)/2,angle=.5*Math.atan2(2*b,a-d);if(l2<=0)continue;
ctx.beginPath();ctx.ellipse(450+p[0]*scale,350-p[1]*scale,Math.sqrt(l1)*scale,Math.sqrt(l2)*scale,-angle,0,Math.PI*2);ctx.strokeStyle='rgba(30,80,130,.2)';ctx.stroke();
ctx.fillStyle='#195080';ctx.fillRect(448+p[0]*scale,348-p[1]*scale,3,3);}
data.axes.forEach((axis,i)=>{ctx.beginPath();ctx.moveTo(450,350);ctx.lineTo(450+dot(axis,ex)*150,350-dot(axis,ey)*150);ctx.strokeStyle=['#d33','#2a5','#2a5'][i];ctx.stroke();ctx.fillText(['N1','P1','P2'][i],450+dot(axis,ex)*160,350-dot(axis,ey)*160)})}
c.onpointerdown=e=>{last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(last){yaw+=(e.clientX-last[0])*.01;pitch+=(e.clientY-last[1])*.01;last=[e.clientX,e.clientY];render()}};c.onpointerup=()=>last=null;render();</script>'''
    output=paths.inside("figures","S7.html");output.write_text(html.replace("PAYLOAD",json.dumps(payload)))
    return {"output":paths.relative(output),"screenshot_note":"Capture viewpoints manually; original manuscript screenshots are not reproduced"}
