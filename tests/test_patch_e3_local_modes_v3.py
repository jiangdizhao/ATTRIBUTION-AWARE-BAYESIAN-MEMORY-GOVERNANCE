"""Scientific invariants for the local multimodal patch E3 audit."""
import unittest
import numpy as np
import torch
from scripts import abmg_patch_e3_local_modes_v3 as m


class PatchLocalE3Tests(unittest.TestCase):
    def test_full_patch_scores_can_see_tail_beyond_top64(self):
        s=m.LocalPatchMemory(np.array([[1.,0.]]),np.zeros(2),np.ones(2))
        patches=np.zeros((784,2))
        patches[600]=[25.,25.]
        self.assertGreater(s.image_score(patches,8),s.image_score(patches[:64],8))

    def test_local_only_updates_matching_mode(self):
        s=m.LocalPatchMemory(np.eye(2),np.zeros(2),np.ones(2),'nig')
        before=s.mu.copy()
        receipt=s.update(np.array([[0.,2.],[0.,3.]]))
        self.assertEqual(receipt['affected_modes'],[1])
        self.assertAlmostEqual(receipt['effective_mass_added'],1.)
        self.assertTrue(np.array_equal(before[0],s.mu[0]))
        self.assertTrue(np.array_equal(s.mass,np.array([0.,1.])))
        self.assertEqual(s.images_seen,1)

    def test_one_image_unit_independent_of_patch_count(self):
        a=m.LocalPatchMemory(np.eye(2),np.zeros(2),np.ones(2))
        b=m.LocalPatchMemory(np.eye(2),np.zeros(2),np.ones(2))
        a.update(np.tile([2.,0.],(32,1)))
        b.update(np.tile([2.,0.],(784,1)))
        self.assertAlmostEqual(a.mass.sum(),1.)
        self.assertAlmostEqual(b.mass.sum(),1.)
        self.assertTrue(np.allclose(a.mu,b.mu))
        self.assertTrue(np.allclose(a.beta,b.beta))

    def test_gaussian_nig_and_prior_stay_finite(self):
        rng=np.random.default_rng(10)
        keys=m.choose_keys(rng.normal(size=(20,16)),8,5)
        self.assertEqual(keys.shape,(8,16))
        for name in m.MODELS:
            s=m.LocalPatchMemory(keys,np.zeros(16),np.ones(16),name)
            x=rng.normal(size=(784,16))
            s.update(x[:32])
            self.assertTrue(np.isfinite(s.surprise(x)).all())
            self.assertEqual(len(s.surprise(x)),784)
            self.assertAlmostEqual(s.mass.sum(),1.)

    def test_invalid_mass_guard(self):
        s=m.LocalPatchMemory(np.eye(2),np.zeros(2),np.ones(2))
        with self.assertRaises(ValueError):
            s.update(np.ones((2,2)),np.ones(2))

    def test_protocol_matching_and_no_sentinel_writes(self):
        rng=np.random.default_rng(2)
        rows=[]; patches=[]
        for cat in ('dev0','dev1','held0','held1'):
            for role,n in (('train_normal',7),('test_normal',4),('defect',3)):
                for j in range(n):
                    x=rng.normal(0,.2,size=(784,8))
                    if role=='defect': x[700:]+=3
                    rows.append({'category':cat,'role':role,'image_id':f'{cat}/{role}/{j}'})
                    patches.append(torch.tensor(x,dtype=torch.float16))
        evidence={'schema':'abmg.e4.evidence.v1','outer_fold':0,
                  'source_cv_groups':[['held0','held1'],[],[],[]],
                  'items':rows,'candidate_features':torch.stack(patches)}
        out=m.run(evidence,0,budget=4,support=2,seed=3,modes=2,patch_write=16)
        self.assertEqual(len(out['results']),2)
        self.assertEqual(out['protocol']['score_patches'],784)
        for row in out['results']:
            self.assertTrue(set(row['query_image_ids']).isdisjoint(row['sentinel_image_ids']))
            self.assertEqual(row['n_queries'],4)
            cp=row['checkpoints'][-1]['branches']
            for name in ('global_gaussian','global_nig','local_gaussian','local_nig'):
                self.assertAlmostEqual(sum(cp[name]['mode_mass']),6.)
                self.assertTrue(0 <= cp[name]['auroc'] <= 1)
        again=m.run(evidence,0,budget=4,support=2,seed=3,modes=2,patch_write=16)
        self.assertEqual(out,again)


if __name__=='__main__':
    unittest.main()
