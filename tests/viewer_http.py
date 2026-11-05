"""Integration check against a temporary localhost server and synthetic fixtures.

Run from the checkout: python tests/viewer_http.py
No stimulus generation or checkpoint inference is performed.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from test_viewer import ViewerContracts
from iclr_repro import paths


def main():
    original=paths.ROOT
    ViewerContracts.setUpClass();root=ViewerContracts.root
    for name in ['compare.html','epochs.html']:
        target=root/'viewers'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(original/'viewers'/name,target)
    (root/'docs').mkdir();shutil.copyfile(original/'docs/VIEWERS.md',root/'docs/VIEWERS.md')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'PYTHONPATH':str(original/'src'),'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
    process=subprocess.Popen([sys.executable,'-c',f'from pathlib import Path; from iclr_repro import paths,viewer; paths.ROOT=Path({str(root)!r}); viewer.serve({port})'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    base=f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(base+'/api/status',timeout=1) as response:assert json.load(response)['dataset_ready']
                break
            except urllib.error.URLError:time.sleep(.1)
        else:raise AssertionError('Server did not start')
        for route in ['/viewers/compare.html','/viewers/epochs.html','/docs/VIEWERS.md']:
            with urllib.request.urlopen(base+route) as r:assert r.status==200 and len(r.read())>100
        for method in ['GET','HEAD']:
            try:urllib.request.urlopen(urllib.request.Request(base+'/configs/protocol.json',method=method));raise AssertionError('Unexpected file access')
            except urllib.error.HTTPError as e:assert e.code==404
        fixture=ViewerContracts();raw=fixture.response().getvalue()
        request=urllib.request.Request(base+'/api/project',data=raw,headers={'Content-Type':'application/octet-stream'},method='POST')
        with urllib.request.urlopen(request,timeout=60) as r:data=json.load(r)
        assert data['schema']=='np-representation/v1' and data['states'][0]['network']=='example'
        assert (root/data['saved_to']).is_file()
        assert not list((root/'results').rglob('*.npz'))
        bad=urllib.request.Request(base+'/api/project',data=b'bad archive',headers={'Content-Type':'application/octet-stream'},method='POST')
        try:urllib.request.urlopen(bad);raise AssertionError('Bad archive accepted')
        except urllib.error.HTTPError as e:assert e.code==400 and json.load(e)['error']
        with urllib.request.urlopen(base+'/api/status') as r:assert json.load(r)['ready']
        print(json.dumps({'http_upload':'pass','saved_projection':data['saved_to'],'raw_response_cache_written':False,'bad_archive_rejected':True,'server_responsive_after_error':True}))
    finally:
        process.terminate();process.wait(timeout=10);process.stderr.close();ViewerContracts.tearDownClass()

if __name__=='__main__':main()
