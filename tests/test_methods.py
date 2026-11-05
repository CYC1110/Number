import importlib
import json
import pkgutil
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import iclr_repro
from iclr_repro import paths
from iclr_repro.geometry import registry,measure,ablate,p_classes,orthonormal_union
from iclr_repro.estimators.basis import principal_angles
from iclr_repro.estimators.bc import bhattacharyya_all_pairs
from iclr_repro.estimators.selectivity import SelectivityConfig,compute_selectivity
from iclr_repro.readouts import train_affine,predict,paired_statistics
from iclr_repro.training import training_plan,simclr_loss,dino_loss
from iclr_repro.fid import frechet_distance
from iclr_repro.figures import collect,network_summary,seed_values,endpoint


def fixture():
    records=[]
    for n in range(1,31):
        for w in range(n+1):
            for regime in [1,2,3]:
                for i in range(6):records.append({"N":n,"w":w,"b":n-w,"set_id":regime,"path":f"N{n}_W{w}_S{regime}_{i}"})
    table,cells,balance=registry(pd.DataFrame(records))
    rng=np.random.default_rng(5);n=table.N.to_numpy();p=table.w.to_numpy()/n
    x=np.column_stack([n,p,p*p,n*p,np.sin(n),np.cos(p)])+.02*rng.normal(size=(len(table),6))
    return x.astype(np.float32),table,cells,balance


class Methods(unittest.TestCase):
    def test_all_imports_without_data_or_model_creation(self):
        for module in pkgutil.walk_packages(iclr_repro.__path__,iclr_repro.__name__+"."):
            importlib.import_module(module.name)

    def test_protocol_and_checkpoint_path_identity(self):
        protocol=paths.protocol()
        original={"untrained","pretrained","label_shuffled","dvd"}
        self.assertEqual(sum(paths.state_counts().values()),145)
        for state in protocol["states"]:
            self.assertEqual(len(paths.runs(state)),20 if state in original else 5)
        for state in original:
            self.assertEqual(paths.runs(state),tuple(f"out_{i:02d}" for i in range(4,24)))
            self.assertEqual([paths.run_seed(state,r) for r in paths.runs(state)],list(range(5962,5979))+[5959,5960,5961])
        with self.assertRaises(ValueError):training_plan("simclr","out_23")
        self.assertEqual(training_plan("pretrained","out_23")["seed"],5961)
        for state in protocol["states"]:
            for run in paths.runs(state):
                plan=training_plan(state,run)
                self.assertEqual(plan["checkpoint"],paths.relative(paths.checkpoint(state,run,plan["epochs"])))
        with self.assertRaises(ValueError):paths.inside("../escape")
        with self.assertRaises(ValueError):paths.checkpoint("../escape","out_04",0)

    def test_p_bins_exact_integer_fractions(self):
        self.assertEqual(p_classes(np.array([12]*8),np.array([0,1,3,5,7,9,11,12])).tolist(),[0,1,2,3,4,5,6,6])

    def test_cell_set_balance_and_disjoint_folds(self):
        x,table,cells,balance=fixture()
        for (cell,fold),group in table.groupby(["cell_id","fold3_id"]):
            self.assertEqual(group.groupby("set_id").size().nunique(),1)
        indices=[set(np.flatnonzero(table.fold3_id.to_numpy()==i)) for i in range(3)]
        self.assertFalse(indices[0]&indices[1] or indices[1]&indices[2] or indices[2]&indices[0])
        self.assertEqual(table.w_ratio.tolist(),(table.w/table.N).tolist())

    def test_bc_matches_analytic_gaussians_and_is_symmetric(self):
        mu=np.array([[0.,0.],[2.,0.]]);cov=np.stack([np.eye(2),np.eye(2)])
        value=bhattacharyya_all_pairs(mu,cov,np.array([0]),np.array([1]))["overlap"][0]
        self.assertAlmostEqual(value,np.exp(-.5),12)
        reverse=bhattacharyya_all_pairs(mu,cov,np.array([1]),np.array([0]))["overlap"][0]
        self.assertEqual(value,reverse)
        cov[1]=0
        with self.assertRaises(RuntimeError):bhattacharyya_all_pairs(mu,cov,np.array([0]),np.array([1]))

    def test_angle_and_union_degeneracy(self):
        un=np.eye(4)[:,:1];up=np.eye(4)[:,1:3]
        self.assertAlmostEqual(principal_angles(un,up,1,2)[0],90)
        with self.assertRaises(ValueError):orthonormal_union(un,np.eye(4)[:,:2])

    def test_geometry_common_scalar_invariance(self):
        x,table,cells,_=fixture()
        first,_,_=measure(x,table,cells);second,_,_=measure(x*3,table,cells)
        self.assertAlmostEqual(first["principal_angle_deg"],second["principal_angle_deg"],places=3)
        self.assertAlmostEqual(first["all_pair_bc"],second["all_pair_bc"],places=5)
        self.assertEqual(len(first["folds"]),3)

    def test_raw_ablation_identity_and_frozen_logits(self):
        x=np.array([[10.,2.,3.],[12.,4.,5.]],dtype=np.float32);u=np.eye(3,dtype=np.float32)[:,:1]
        changed=ablate(x,u)
        np.testing.assert_array_equal(changed[:,0],0);np.testing.assert_array_equal(ablate(changed,u),changed)
        weight=np.arange(6,dtype=float).reshape(3,2);bias=np.ones(2)
        np.testing.assert_allclose(changed@weight+bias,(x@weight+bias)-(x@u)@(u.T@weight))

    def test_anova_balanced_quantity_signal(self):
        n=np.repeat(np.arange(1,5),60);s=np.tile(np.repeat(np.arange(1,4),20),4)
        rng=np.random.default_rng(4);x=np.column_stack([n+rng.normal(0,.05,len(n)),rng.normal(size=len(n))])
        result=compute_selectivity(x,n,s,SelectivityConfig())
        self.assertLess(result["p_num"][0],.01);self.assertGreater(result["p_set"][0],.01)

    def test_decoder_train_validation_guard(self):
        x=np.array([[0.,1.],[1.,0.],[0.,2.],[2.,0.]],np.float32);y=np.array([0,1,0,1])
        model,selection=train_affine(x,y,np.array([0,1]),np.array([2,3]),seed=3,epochs=3)
        self.assertTrue(1<=selection["selected_epoch"]<=3);self.assertEqual(len(predict(model,x)),4)
        with self.assertRaises(ValueError):train_affine(x,y,np.array([0,1]),np.array([1,2]),seed=3,epochs=1)

    def test_losses_on_small_embedding_arrays(self):
        torch.manual_seed(1);a=torch.randn(3,4,requires_grad=True);b=torch.randn(3,4,requires_grad=True)
        loss=simclr_loss(a,b);self.assertTrue(torch.isfinite(loss));loss.backward()
        students=[torch.randn(2,5,requires_grad=True) for _ in range(4)];teachers=[torch.randn(2,5) for _ in range(2)]
        loss=dino_loss(students,teachers,torch.zeros(1,5));self.assertTrue(torch.isfinite(loss));loss.backward()

    def test_fid_analytic_translation(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(20,3))
        self.assertAlmostEqual(frechet_distance(x,x),0,places=7)
        self.assertAlmostEqual(frechet_distance(x,x+np.array([2.,0.,0.])),4,places=7)

    def test_paired_original_twenty_and_reordered_five(self):
        for count in [20,5]:
            before={s:float(i) for i,s in enumerate(range(100,100+count))}
            after={s:before[s]+(s-99)*.1 for s in reversed(list(before))}
            stats=paired_statistics(before,after)
            self.assertEqual((stats["n_before"],stats["n_after"],stats["n_pairs"],stats["df"]),(count,count,count,count-1))
            self.assertGreater(stats["t"],0)
            self.assertAlmostEqual(stats["mean_difference_after_minus_before"],(count+1)*.05)

    def test_unequal_cohorts_pair_explicit_shared_seeds(self):
        before={s:float(i) for i,s in enumerate(range(100,120))}
        after={s:before[s]+(s-99)*.2 for s in [104,100,103,101,102]}
        stats=paired_statistics(before,after)
        self.assertEqual((stats["n_before"],stats["n_after"],stats["n_pairs"],stats["df"]),(20,5,5,4))
        self.assertEqual(stats["paired_seeds"],list(range(100,105)))
        self.assertEqual(stats["unmatched_before_seeds"],list(range(105,120)))
        self.assertAlmostEqual(stats["mean_difference_after_minus_before"],.6)
        overlap=paired_statistics({1:1,2:2,3:3},{2:4,3:6,4:8})
        self.assertEqual(overlap["paired_seeds"],[2,3])
        with self.assertRaises(ValueError):paired_statistics({1:1,2:2},{3:3,4:4})
        with self.assertRaises(ValueError):paired_statistics([1,2],[2,3])
        with self.assertRaises(ValueError):paired_statistics({1:1,"1":2},{1:3,2:4})
        with self.assertRaises(ValueError):paired_statistics({1:1,2:float("nan")},{1:2,2:3})

    def test_network_sem_uses_actual_twenty_or_five(self):
        for count in [20,5]:
            values=np.arange(count,dtype=float);stats=network_summary(values)
            self.assertEqual(stats["n"],count)
            self.assertAlmostEqual(stats["sem"],np.std(values,ddof=1)/np.sqrt(count))
        with self.assertRaises(ValueError):network_summary([1,np.nan])

    def test_collect_complete_condition_specific_cohorts_and_provenance(self):
        protocol_text=paths.inside("configs","protocol.json").read_text()
        with tempfile.TemporaryDirectory() as directory,patch.object(paths,"ROOT",Path(directory)):
            paths.inside("configs").mkdir();paths.inside("configs","protocol.json").write_text(protocol_text)
            for state in ["pretrained","simclr"]:
                for run in paths.runs(state):
                    seed=paths.run_seed(state,run)
                    paths.write_json(paths.result(state,run,endpoint(state))/"metrics.json",
                        {"state":state,"run":run,"epoch":endpoint(state),"seed":seed,
                         "expected_networks_in_state":len(paths.runs(state)),"basis_scope":"full",
                         "source":{"seed":seed,"dataset_manifest_sha256":"test-fixture"},"measurement":float(seed)})
            records=collect(["pretrained","simclr"])
            self.assertEqual([len(records[s]) for s in ["pretrained","simclr"]],[20,5])
            self.assertEqual(len(seed_values(records,"pretrained","measurement")),20)
            file=paths.result("pretrained","out_23",90)/"metrics.json"
            original=file.read_text();row=json.loads(original);row["seed"]+=1;file.write_text(json.dumps(row))
            with self.assertRaises(ValueError):collect(["pretrained"])
            file.write_text(original);file.unlink()
            with self.assertRaises(FileNotFoundError):collect(["pretrained"])


if __name__=="__main__":unittest.main()
