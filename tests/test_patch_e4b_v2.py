"""Synthetic invariants for source-enriched patch E4B."""
import unittest
import numpy as np
import torch

from scripts import abmg_patch_e4b_v2 as m


class TestPatchE4B(unittest.TestCase):
    def test_mass_and_shuffled_source(self):
        base={d:np.eye(4)[i] for i,d in enumerate(m.CORE4)}
        labels={'AK':'HS','HS':'AK','QS':'ZW','ZW':'QS'}
        memories={p:m.SourceMemory(base) for p in m.POLICIES}
        for p in m.POLICIES: memories[p].update(base['AK'],'AK',p,labels)
        for p in ('broadcast','correct_source','shuffled_source'):
            self.assertAlmostEqual(sum(memories[p].n.values()),5.)
        self.assertEqual(memories['frozen'].n['AK'],1.)
        self.assertEqual(memories['correct_source'].n['AK'],2.)
        self.assertEqual(memories['shuffled_source'].n['HS'],2.)

    def test_fixed_sentinel_and_replay(self):
        rng=np.random.default_rng(4)
        rows=[];patches=[]
        for cat in ('dev0','dev1','held0','held1'):
            for role,n in [('train_normal',6),('test_normal',4),('defect',16)]:
                for k in range(n):
                    d=m.CORE4[k%4] if role=='defect' else None
                    x=rng.normal(0,.12,(784,16))
                    if d is not None: x[:64,(k%4)*4:(k%4+1)*4]+=1
                    rows.append({'category':cat,'role':role,'image_id':f'{cat}_{role}_{k}',
                                 'defect_source_offline_only':d})
                    patches.append(torch.tensor(x,dtype=torch.float16))
        a={'schema':'abmg.e4.evidence.v1','outer_fold':0,
           'source_categories':['dev0','dev1','held0','held1'],
           'source_cv_groups':[['held0','held1'],[],[],[]],
           'items':rows,'candidate_features':torch.stack(patches)}
        result=m.run(a,0,8,1,2)
        self.assertEqual(result['queries_realized'],8)
        self.assertTrue(set(result['query_image_ids']).isdisjoint(result['sentinel_image_ids']))
        self.assertEqual(result['checkpoints'][0]['queries'],0)
        self.assertEqual(result['checkpoints'][-1]['queries'],8)
        self.assertEqual(len(result['checkpoints'][-1]['branches']),4)


if __name__=='__main__':
    unittest.main()
