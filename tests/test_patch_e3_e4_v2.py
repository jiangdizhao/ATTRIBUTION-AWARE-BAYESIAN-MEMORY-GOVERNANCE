"""Focused synthetic tests for full-patch E3 / E4A mechanism validation."""
import unittest
import numpy as np
import torch

from scripts import abmg_patch_e3_e4_v2 as m


class PatchE3E4Tests(unittest.TestCase):
    def test_per_image_mass_not_per_patch(self):
        s = m.PatchState(np.zeros(16), np.ones(16), groups=8)
        s.update(np.ones((32, 16)), np.ones(8)/8)
        self.assertTrue(np.allclose(s.mass, [0.125]*8))
        self.assertTrue(np.allclose(s.kappa, 0.175))

    def test_fractional_nig_predictive_is_finite(self):
        s = m.PatchState(np.zeros(16), np.ones(16), groups=8)
        s.update(np.ones((4, 16)), np.ones(8))
        self.assertEqual(s.surprise(np.ones((4, 16))).shape, (4, 8))
        self.assertTrue(np.isfinite(s.surprise(np.ones((4, 16)))).all())
        self.assertTrue(np.all(s.beta > 0))

    def test_gaussian_differs_from_nig(self):
        a = m.PatchState(np.zeros(16), np.ones(16), 'gaussian', 8)
        b = m.PatchState(np.zeros(16), np.ones(16), 'nig', 8)
        for s in (a, b):
            s.update(np.full((8, 16), .3), np.ones(8))
        self.assertFalse(np.allclose(a.surprise(np.ones((8, 16))),
                                         b.surprise(np.ones((8, 16)))))

    def test_credit_mass_and_write_location(self):
        s = m.PatchState(np.zeros(16), np.ones(16), groups=8)
        p = np.tile(np.arange(16, dtype=float), (8,1))
        perm = np.roll(np.arange(8),1)
        b=m.choose_credit(s,p,'broadcast',perm)
        c=m.choose_credit(s,p,'selective',perm)
        d=m.choose_credit(s,p,'shuffled',perm)
        self.assertAlmostEqual(b.sum(),1.)
        self.assertAlmostEqual(c.sum(),1.)
        self.assertAlmostEqual(d.sum(),1.)
        self.assertEqual(np.count_nonzero(c),8)
        self.assertEqual(np.count_nonzero(d),8)
        self.assertFalse(np.array_equal(c,d))
        self.assertTrue(np.array_equal(m.choose_credit(s,p,'frozen',perm),
                                       np.zeros(8)))

    def test_cv_and_sentinel_separation(self):
        rng=np.random.default_rng(9)
        items=[]; patches=[]
        groups=[['held0','held1'],[],[],[]]
        for cat in ['dev0','dev1','held0','held1']:
            for role,n in [('train_normal',12),('test_normal',4),('defect',3)]:
                for k in range(n):
                    items.append({'category':cat,'role':role,'image_id':f'{cat}/{role}/{k}'})
                    x=rng.normal(0,.2,(784,16))
                    if role=='defect': x[:64]+=3
                    patches.append(torch.tensor(x,dtype=torch.float16))
        a={'schema':'abmg.e4.evidence.v1','outer_fold':0,
           'source_cv_groups':groups,'items':items,
           'candidate_features':torch.stack(patches)}
        result=m.run(a,'e4a',0,8,3,2,16)
        self.assertEqual(len(result['results']),2)
        for row in result['results']:
            self.assertEqual(row['n_queries'],8)
            self.assertEqual(row['n_sentinel_normal'],2)
            self.assertTrue(set(row['queried_image_ids']).isdisjoint(row['sentinel_image_ids']))
            self.assertEqual(len(set(row['queried_image_ids'])),len(row['queried_image_ids']))
            br=row['checkpoints'][-1]['branches']
            for key in ('broadcast','selective','shuffled'):
                self.assertAlmostEqual(sum(br[key]['effective_mass']),24)
            self.assertAlmostEqual(sum(br['frozen']['effective_mass']),16)
        e3=m.run(a,'e3',0,8,3,2,16)
        self.assertTrue(all('gaussian' in x['checkpoints'][-1]['branches']
                            for x in e3['results']))


if __name__ == '__main__':
    unittest.main()
