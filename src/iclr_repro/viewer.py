"""Portable N/P viewers and local response import. No raw response cache is written."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from . import paths
from .geometry import registry, cell_means, conditioned7, gaussian_cells
from .estimators.basis import conditioned_bases

SCHEMA = 'np-representation/v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dataset_contract(manifest='data/np/metadata.json'):
    path = paths.inside(manifest)
    if not path.is_file():
        raise ValueError('Place the published NP dataset and metadata.json in data/np first. Bundled viewers work without it.')
    raw = path.read_bytes()
    table = pd.DataFrame(json.loads(raw)['images'])
    required = {'path', 'N', 'w', 'set_id', 'sha256'}
    if not required <= set(table):
        raise ValueError(f'Metadata requires columns {sorted(required)}')
    if table.path.duplicated().any() or table.N.lt(0).any():
        raise ValueError('Duplicate image paths or negative N in metadata')
    for value in table.path:
        p = Path(value)
        if p.is_absolute() or '..' in p.parts or '\\' in value:
            raise ValueError('Image paths must be relative to data/np')
    table = table.loc[table.N.gt(0)].reset_index(drop=True)
    if set(table.N) != set(range(1, 31)):
        raise ValueError('NP stimuli must cover N=1 through 30')
    if not table.sha256.str.fullmatch('[0-9a-f]{64}').all():
        raise ValueError('Every stimulus needs its published SHA-256')
    table, cells, _ = registry(table, paths.protocol()['split_seed'])
    if len(cells) != 495:
        raise ValueError('NP stimuli must cover all 495 nonblank (N,w) cells')
    return table, cells, digest(raw)


def template(output='data/responses/template', manifest='data/np/metadata.json'):
    table, _, sha = dataset_contract(manifest)
    target = paths.inside(output); target.mkdir(parents=True, exist_ok=True)
    table[['path','sha256','N','w','b','set_id','fold3_id']].to_csv(target/'image_order.csv', index=False)
    np.savez_compressed(target/'response_template.npz', responses=np.empty((len(table), 0), np.float32),
                        image_paths=table.path.to_numpy(dtype=str), dataset_manifest_sha256=sha,
                        network_id='my_network', run='run_01', label='My network', epoch=90)
    return {'template': paths.relative(target/'response_template.npz'), 'images': len(table),
            'dataset_manifest_sha256': sha, 'next': 'Replace responses with finite [images, features] activations. Keep image_paths aligned.'}


def scalar_string(data, key, default=None):
    if key not in data:
        if default is not None: return default
        raise ValueError(f'Missing response field: {key}')
    a = np.asarray(data[key])
    if a.ndim != 0 or a.dtype.kind not in 'US':
        raise ValueError(f'{key} must be a Unicode string scalar (no pickle/object arrays)')
    return str(a.item())


def activation_scalar(x, table, cells):
    """Canonical equal-N scale over 406 primary cells, all assigned folds.

    Scalar display normalization only; the N/P basis still fits B+C exclusively.
    """
    support = cells[(cells.w >= 1) & (cells.b >= 1) & (cells.N <= 29)]
    means=[]; squares=[]
    ids=table.cell_id.to_numpy(); assigned=table.fold3_id.to_numpy() >= 0
    for cell in support.itertuples():
        values=np.asarray(x[(ids==cell.cell_id)&assigned], dtype=np.float64)
        means.append(values.mean(0)); squares.append(np.einsum('ij,ij->',values,values)/len(values))
    means=np.asarray(means); weights=1/(28*(support.N.to_numpy()-1))
    mean=weights @ means
    variance=np.sum(weights*(np.asarray(squares)-2*(means@mean)+np.dot(mean,mean)))/x.shape[1]
    if not np.isfinite(variance) or variance <= 1e-12: raise ValueError('Activation scale is undefined')
    return float(np.sqrt(variance))


def project_response(source, manifest='data/np/metadata.json', p_bins=6):
    table, cells, sha = dataset_contract(manifest)
    # Reject archives that expand beyond the documented in-memory import limit.
    with zipfile.ZipFile(source) as archive:
        if sum(i.file_size for i in archive.infolist()) > 8 * 1024**3:
            raise ValueError('Uncompressed response exceeds the 8 GiB importer limit')
    if hasattr(source, 'seek'): source.seek(0)
    with np.load(source, allow_pickle=False) as data:
        if scalar_string(data, 'dataset_manifest_sha256') != sha:
            raise ValueError('Dataset manifest SHA-256 mismatch; use the template for this exact dataset')
        network = paths.identifier(scalar_string(data, 'network_id'))
        run = paths.identifier(scalar_string(data, 'run', 'run_01'))
        label = scalar_string(data, 'label', network)
        if len(label) > 200: raise ValueError('Label must be at most 200 characters')
        ep = np.asarray(data['epoch'])
        if ep.ndim or ep.dtype.kind not in 'iu' or int(ep) < 0: raise ValueError('epoch must be a nonnegative integer scalar')
        epoch = int(ep)
        names = np.asarray(data['image_paths'])
        if names.ndim != 1 or names.dtype.kind not in 'US': raise ValueError('image_paths must be a Unicode string vector')
        names = names.astype(str)
        if len(set(names)) != len(names): raise ValueError('Duplicate image paths in responses')
        if set(names) != set(table.path): raise ValueError('Responses must contain every nonblank image exactly once; paths do not match metadata')
        x = np.asarray(data['responses'])
        if x.dtype.kind not in 'fiu' or x.ndim != 2 or x.shape[0] != len(names) or x.shape[1] < 3:
            raise ValueError('responses must be a numeric [images, features>=3] matrix; fill the empty template first')
        if not np.isfinite(x).all(): raise ValueError('Responses contain NaN or infinity')
        order = {name:i for i,name in enumerate(names)}
        indices=np.array([order[p] for p in table.path])
        if not np.array_equal(indices,np.arange(len(indices))): x=x[indices]
        x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('Responses exceed float32 range')
    assignment = table.fold3_id.to_numpy()
    train = np.flatnonzero((assignment == 1) | (assignment == 2)); held = np.flatnonzero(assignment == 0)
    mu, count = cell_means(x, table, cells, train)
    fit = conditioned_bases(mu, count, cells) if p_bins == 6 else conditioned7(mu, count, cells)
    if fit['N'].singular_values[0] <= 1e-10 or fit['P'].singular_values[1] <= 1e-10:
        raise ValueError('N1/P2 is undefined: insufficient quantity-dependent variation')
    n = fit['N'].basis[:,0].astype(float); n /= np.linalg.norm(n)
    # Deterministic sign: N response increases with numerosity in training data.
    scores = x[train] @ n; cov = np.dot(scores-scores.mean(), table.N.to_numpy()[train]-table.N.to_numpy()[train].mean())
    if cov < 0: n *= -1
    p, r = np.linalg.qr(fit['P'].basis[:,:2].astype(float)); p *= np.where(np.diag(r) >= 0, 1., -1.)
    v = p @ (p.T @ n); cosine = np.linalg.norm(v)
    if cosine < 1e-8 or 1-cosine < 1e-10:
        raise ValueError('Principal-pair 3D frame is degenerate (N is perpendicular to, or lies in, the P plane)')
    v /= cosine; normal = n-v*np.dot(n,v); normal /= np.linalg.norm(normal)
    y = p @ np.array([-np.dot(p[:,1], v), np.dot(p[:,0], v)])
    y /= np.linalg.norm(y)
    q = np.column_stack([v,y,normal])
    if not np.allclose(q.T@q, np.eye(3), atol=2e-6): raise ValueError('Could not construct an orthonormal frame')
    scale = activation_scalar(x, table, cells)
    z = x[held] @ q
    del x
    z -= z.mean(axis=0)
    test = table.iloc[held]; ids = test.cell_id.to_numpy()
    cell_mu, cell_cov = gaussian_cells(z, ids, len(cells))
    nc = np.bincount(ids, minlength=len(cells))
    # Display uses 6x6 colors independently of P6/P7 analysis binning.
    nbin = np.clip((cells.N.to_numpy()-1)//5, 0, 5)
    pbin = np.clip(np.digitize(cells.w_ratio, np.linspace(0,1,7)[1:-1]), 0, 5)
    means=[]; covs=[]; bins=[]; counts=[]
    for a in range(6):
        for b in range(6):
            chosen = np.flatnonzero((nbin==a)&(pbin==b)); weight=nc[chosen]
            m = np.average(cell_mu[chosen], axis=0, weights=weight); delta=cell_mu[chosen]-m
            c = np.average(cell_cov[chosen]+delta[:,:,None]*delta[:,None,:], axis=0, weights=weight)
            means.append(m.tolist()); covs.append(c.tolist()); bins.append([a,b]); counts.append(int(weight.sum()))
    point_bins = np.column_stack([np.clip((test.N.to_numpy()-1)//5,0,5), np.clip(np.digitize(test.w_ratio,np.linspace(0,1,7)[1:-1]),0,5)])
    start = z[test.N.to_numpy()==1].mean(axis=0)
    state = dict(id=f'{network}/{run}/{epoch}', network=network, run=run, label=label, epoch=epoch,
                 dataset=f'NP {sha[:12]}', datasetHash=sha, protocol=f'P{p_bins} · B+C basis / held-out A · exact-cell LW pooled',
                 theta=float(np.degrees(np.arccos(np.clip(cosine,0,1)))), means=means, covariances=covs,
                 pBasis=(q.T@p).tolist(), startPoint=start.tolist(), startPointCount=int((test.N==1).sum()),
                 bins=bins, counts=counts, pointBins=point_bins.tolist(),
                 pointsF32=base64.b64encode(z.astype('<f4').tobytes()).decode(),
                 sourceUnits='raw response units', scalar=scale, projectionRMS=float(np.sqrt(np.mean(np.sum(z*z,axis=1)))),
                 source={'dataset_manifest':manifest,'dataset_manifest_sha256':sha,'feature_dimension':int(q.shape[0]),
                         'basis_scope':'B+C','heldout_fold':'A','normalization':'equal-N, 406 primary cells, assigned A+B+C; scalar display normalization only', 'raw_activations_stored':False})
    return {'schema':SCHEMA, 'states':[state]}


def write_bundle(bundle, output=None):
    state=bundle['states'][0]
    path=paths.inside(output) if output else paths.inside('results',state['network'],state['run'],f"epoch_{state['epoch']:03d}",'representation.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False)+'\n')
    return path


def build():
    asset=paths.inside('viewers','assets')
    config=json.loads(paths.inside('configs','viewers.json').read_text())
    reports=[]
    for mode, entry in config['pages'].items():
        states=[]
        for filename in entry['bundles']:
            bundle=json.loads(paths.inside(filename).read_text())
            if bundle.get('schema')!=SCHEMA: raise ValueError(f'Unknown projection schema: {filename}')
            states.extend(bundle['states'])
        ids=[s['id'] for s in states]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate state identities in build inputs')
        payload={'schema':SCHEMA,'mode':mode,'states':states,'palette':config['palette']}
        raw=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
        from html import escape
        replacements={'TITLE':escape(entry['title']),'CSS':(asset/'viewer.css').read_text(),'DATA':raw,
                      'JS':(asset/'viewer.js').read_text(),'GIF_JS':(asset/'gif-worker.js').read_text()}
        html=re.sub(r'__(TITLE|CSS|DATA|JS|GIF_JS)__',lambda m:replacements[m[1]],(asset/'viewer.html.in').read_text())
        target=paths.inside('viewers',entry['output']);target.write_text(html)
        reports.append({'path':paths.relative(target),'states':len(states),'sha256':digest(target.read_bytes())})
    paths.write_json(paths.inside('viewers','build_manifest.json'),{'pages':reports,'normalization':'equal-N, 406 primary cells, assigned A+B+C; scalar display normalization only', 'raw_activations_stored':False})
    return reports


def serve(port=8765):
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import urlsplit, unquote
    import threading
    lock=threading.Lock()
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw): super().__init__(*a,directory=str(paths.ROOT),**kw)
        def allowed(self):
            host=self.headers.get('Host','')
            origin=self.headers.get('Origin')
            return host in {f'127.0.0.1:{port}',f'localhost:{port}'} and (origin is None or origin==f'http://{host}')
        def do_GET(self):
            if not self.allowed(): return self.send_error(403)
            route=unquote(urlsplit(self.path).path)
            if route=='/api/status':
                return self.send_json({'ready':True,'dataset_ready':paths.inside('data/np/metadata.json').is_file()})
            if route=='/': self.path='/viewers/compare.html'
            target=(paths.ROOT/unquote(urlsplit(self.path).path).lstrip('/')).resolve()
            if (not target.is_relative_to(paths.inside('viewers')) and target != paths.inside('docs/VIEWERS.md')) or not target.is_file(): return self.send_error(404)
            super().do_GET()
        def do_HEAD(self):
            if not self.allowed(): return self.send_error(403)
            target=(paths.ROOT/unquote(urlsplit(self.path).path).lstrip('/')).resolve()
            if (not target.is_relative_to(paths.inside('viewers')) and target != paths.inside('docs/VIEWERS.md')) or not target.is_file(): return self.send_error(404)
            super().do_HEAD()
        def send_json(self,obj,status=200):
            data=json.dumps(obj,separators=(',',':'),allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if not self.allowed() or self.path!='/api/project' or self.headers.get('Content-Type')!='application/octet-stream': return self.send_error(403)
            try: size=int(self.headers.get('Content-Length','0'))
            except ValueError: return self.send_json({'error':'Invalid upload length'},400)
            if size<=0 or size>4*1024**3: return self.send_json({'error':'Upload limit is 4 GiB compressed; use the CLI for larger files'},413)
            if not lock.acquire(blocking=False): return self.send_json({'error':'Another response is being converted; wait and retry'},409)
            try:
                raw=self.rfile.read(size)
                if len(raw)!=size: raise ValueError('Incomplete upload')
                p_bins=int(self.headers.get('X-P-Bins','6'))
                if p_bins not in [6,7]: raise ValueError('P bins must be 6 or 7')
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=4): bundle=project_response(io.BytesIO(raw),p_bins=p_bins)
                del raw
                target=write_bundle(bundle)
                bundle['saved_to']=paths.relative(target)
                self.send_json(bundle)
            except (ValueError,KeyError,TypeError,OSError,zipfile.BadZipFile) as error:
                self.send_json({'error':str(error)},400)
            except Exception:
                self.send_json({'error':'Projection failed; inspect the server terminal'},500)
                import traceback; traceback.print_exc()
            finally: lock.release()
    print(f'Open http://127.0.0.1:{port}/viewers/compare.html or /viewers/epochs.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('template');p.add_argument('--output',default='data/responses/template');p.add_argument('--manifest',default='data/np/metadata.json')
    p=sub.add_parser('project');p.add_argument('responses',nargs='+');p.add_argument('--manifest',default='data/np/metadata.json');p.add_argument('--p-bins',type=int,choices=[6,7],default=6)
    p=sub.add_parser('extract');p.add_argument('state');p.add_argument('--run',default='out_04');p.add_argument('--epochs',type=int,nargs='+',required=True);p.add_argument('--device',default='cpu');p.add_argument('--p-bins',type=int,choices=[6,7],default=6)
    sub.add_parser('build');p=sub.add_parser('serve');p.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(argv)
    if args.command=='template': result=template(args.output,args.manifest)
    elif args.command=='project':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            result=[paths.relative(write_bundle(project_response(paths.inside(f),args.manifest,args.p_bins))) for f in args.responses]
    elif args.command=='extract':
        from .pipeline import load_metadata,extract_features
        table=load_metadata('np'); sha=digest(paths.inside('data/np/metadata.json').read_bytes()); result=[]
        for epoch in args.epochs:
            x,proof=extract_features(args.state,args.run,epoch,table,device=args.device)
            buffer=io.BytesIO()
            np.savez(buffer,responses=x,image_paths=table.path.to_numpy(dtype=str),dataset_manifest_sha256=sha,network_id=args.state,run=args.run,label=args.state,epoch=epoch)
            del x;buffer.seek(0)
            bundle=project_response(buffer,p_bins=args.p_bins);buffer.close();bundle['states'][0]['source'].update(proof)
            result.append(paths.relative(write_bundle(bundle)))
    elif args.command=='build': result=build()
    else: return serve(args.port)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
