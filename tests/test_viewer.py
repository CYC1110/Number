"""Response alignment, provenance, subspace geometry, and static build checks."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from iclr_repro import paths, viewer
from iclr_repro.geometry import cell_means, conditioned7
from iclr_repro.estimators.basis import conditioned_bases, principal_angles

class ViewerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        (cls.root/'data/np').mkdir(parents=True);(cls.root/'configs').mkdir()
        (cls.root/'configs/protocol.json').write_bytes(paths.inside('configs/protocol.json').read_bytes())
        rows=[]
        for n in range(1,31):
            for w in range(n+1):
                for s in [1,2,3]:
                    for i in range(6):rows.append(dict(N=n,w=w,set_id=s,path=f'set_{s}/N_{n}/w_{w}_b_{n-w}/{i}.png',sha256='a'*64))
        raw=json.dumps({'images':rows}).encode();(cls.root/'data/np/metadata.json').write_bytes(raw)
        cls.sha=viewer.digest(raw);cls.names=np.array([r['path'] for r in rows]);n=np.array([r['N'] for r in rows]);p=np.array([r['w']/r['N'] for r in rows]);rng=np.random.default_rng(7)
        cls.x=(np.column_stack([n,p,p*p,n*p,np.sin(n),np.cos(p),n*.4+p])+rng.normal(0,.04,(len(rows),7))).astype(np.float32)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def response(self,**changes):
        data=dict(responses=self.x,image_paths=self.names,dataset_manifest_sha256=self.sha,network_id='example',run='seed_7',label='Fixture network',epoch=90);data.update(changes)
        file=io.BytesIO();np.savez_compressed(file,**data);file.seek(0);return file
    def test_projection_alignment_angle_and_origin(self):
        with patch.object(paths,'ROOT',self.root):
            a=viewer.project_response(self.response());s=a['states'][0]
            order=np.random.default_rng(5).permutation(len(self.x));b=viewer.project_response(self.response(responses=self.x[order],image_paths=self.names[order]))
            self.assertEqual(a,b)
            table,cells,_=viewer.dataset_contract();train=np.flatnonzero(table.fold3_id.isin([1,2]));m,c=cell_means(self.x,table,cells,train);fit=conditioned_bases(m,c,cells)
            self.assertAlmostEqual(s['theta'],principal_angles(fit['N'].basis,fit['P'].basis,1,2)[0],places=5)
            z=np.frombuffer(viewer.base64.b64decode(s['pointsF32']),dtype='<f4').reshape(-1,3);held=table[table.fold3_id==0]
            np.testing.assert_allclose(s['startPoint'],z[held.N.to_numpy()==1].mean(0),atol=2e-5)
            self.assertEqual(sum(s['counts']),len(z));self.assertTrue((np.linalg.eigvalsh(s['covariances'])>0).all());self.assertNotIn('responses',s)
            target=viewer.write_bundle(a);self.assertTrue(target.is_relative_to(self.root/'results/example/seed_7/epoch_090'))
            self.assertFalse(any(p.name.endswith('.npz') for p in (self.root/'results').rglob('*')))
            self.fixture=a
    def test_bad_inputs_and_P7(self):
        with patch.object(paths,'ROOT',self.root):
            for change in [dict(dataset_manifest_sha256='wrong'),dict(image_paths=np.repeat(self.names[:1],len(self.names))),dict(responses=self.x[:10]),dict(responses=np.zeros_like(self.x)),dict(responses=np.full_like(self.x,np.nan))]:
                with self.assertRaises(ValueError):viewer.project_response(self.response(**change))
            p7=viewer.project_response(self.response(),p_bins=7)['states'][0]
            table,cells,_=viewer.dataset_contract();train=np.flatnonzero(table.fold3_id.isin([1,2]));m,c=cell_means(self.x,table,cells,train);fit=conditioned7(m,c,cells)
            self.assertAlmostEqual(p7['theta'],principal_angles(fit['N'].basis,fit['P'].basis,1,2)[0],places=5)
            self.assertTrue(p7['protocol'].startswith('P7'))
            scaled=viewer.project_response(self.response(responses=self.x*3),p_bins=7)['states'][0]
            self.assertAlmostEqual(scaled['scalar']/p7['scalar'],3,places=5)
            np.testing.assert_allclose(np.array(scaled['means'])/scaled['scalar'],np.array(p7['means'])/p7['scalar'],atol=2e-4)
    def test_template_and_source_contract(self):
        with patch.object(paths,'ROOT',self.root):
            out=viewer.template();self.assertEqual(out['images'],len(self.x))
            with np.load(self.root/out['template'],allow_pickle=False) as d:
                self.assertEqual(d['responses'].shape,(len(self.x),0));np.testing.assert_array_equal(d['image_paths'],self.names)
        for name in ['compare.html','epochs.html']:
            html=paths.inside('viewers',name).read_text();self.assertNotIn(str(Path.home()),html);self.assertNotIn('__DATA__',html);self.assertNotIn('<script src=',html)

if __name__=='__main__':unittest.main()
