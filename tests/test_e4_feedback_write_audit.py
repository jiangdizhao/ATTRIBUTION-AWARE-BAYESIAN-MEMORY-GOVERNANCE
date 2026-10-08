"""E4 write-location and feedback-integrity unit tests (no Real-IAD/GPU required)."""
from __future__ import annotations

import copy
from collections import Counter
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Core library imported after inserting the same scripts directory used by CLI.
SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    import abmg_e4_feedback_write_audit as e4
except ImportError:
    # The local development-only test workspace may not contain E3 dependencies;
    # a second explicit test run on the real repository is required before claims
    # of integration success. No source files are modified by these stubs.
    raise


class DummyNIG:
    def __init__(self, dim=3):
        self.mu=np.zeros(dim);self.kappa=np.ones(dim);self.alpha=np.ones(dim)*2
        self.beta=np.ones(dim)
        self.count=0

    def update(self,x,weight=1.,*,robust=False):
        self.mu=self.mu+np.asarray(x)*weight
        self.kappa=self.kappa+weight
        self.alpha=self.alpha+weight/2
        self.beta=self.beta+np.asarray(x)**2*weight
        self.count+=1

    def predictive_log_prob(self,x):
        return -np.linalg.norm(np.asarray(x)-self.mu,axis=1)


class E4Tests(unittest.TestCase):
    def setUp(self):
        self.x=np.asarray([1.,0.,0.]); self.source=e4.CORE4[0]
        self.mem={g:{'n':8.,'sum':np.zeros(3)} for g in e4.CORE4}
        self.prior={'mu0':np.zeros(3),'var':np.ones(3)}
        self.perm={g:e4.CORE4[(j+1)%len(e4.CORE4)] for j,g in enumerate(e4.CORE4)}

    def state(self,hard=False,kind='binary'):
        return e4.BranchState(hard,kind,{'product':DummyNIG()},copy.deepcopy(self.mem),{},Counter(),Counter())

    def test_credit_fallback_then_correct_and_shuffled(self):
        r=np.asarray([.7,.1,.1,.1]);h=e4.SourceCentroids.empty()
        np.testing.assert_array_equal(e4.defect_credit('correct',r,self.source,h,self.perm,.5),r)
        h.add(self.source,np.asarray([.1,.7,.1,.1]))
        np.testing.assert_allclose(e4.defect_credit('correct',r,self.source,h,self.perm,.5),[.4,.4,.1,.1])
        np.testing.assert_allclose(e4.defect_credit('shuffled',r,self.source,h,self.perm,.5),r)
        h.add(self.perm[self.source],np.asarray([.1,.1,.7,.1]))
        np.testing.assert_allclose(e4.defect_credit('shuffled',r,self.source,h,self.perm,.5),[.4,.1,.4,.1])
        self.assertEqual(e4.defect_credit('oracle',r,self.source,h,self.perm,.5)[0],1.)
        self.assertEqual(h.count[self.source],1)

    def test_exact_defect_nig_protection_and_fractional_credit(self):
        s=self.state(False,'binary');before=s.fingerprint_nig('product')
        r=np.ones(len(e4.CORE4))/len(e4.CORE4)
        receipt=e4.apply_verified_feedback(s,is_defect=True,is_false_alarm=True,
            category='product',vector_raw=self.x,top8_patches=np.repeat(self.x[None,:],8,axis=0),
            source=self.source,responsibility=r,history=e4.SourceCentroids.empty(),
            permutation=self.perm,eta=.5)
        self.assertEqual(before,s.fingerprint_nig('product'))
        self.assertTrue(receipt['factor_write'])
        self.assertEqual(receipt['normal_write'],False)
        for g in e4.CORE4:
            self.assertAlmostEqual(s.router_memory[g]['n'],8.25)
        self.assertEqual(sum(receipt['factor_credit']),1.0)

    def test_true_positive_only_updates_source_tracker_outside_write(self):
        s=self.state(False,'correct');before=copy.deepcopy(s.router_memory)
        r=np.ones(len(e4.CORE4))/len(e4.CORE4)
        receipt=e4.apply_verified_feedback(s,is_defect=True,is_false_alarm=False,
            category='product',vector_raw=self.x,top8_patches=np.repeat(self.x[None,:],8,axis=0),
            source=self.source,responsibility=r,history=e4.SourceCentroids.empty(),
            permutation=self.perm,eta=.5)
        self.assertEqual(before.keys(),s.router_memory.keys())
        for g in e4.CORE4:
            np.testing.assert_array_equal(before[g]['sum'],s.router_memory[g]['sum'])
        self.assertFalse(receipt['factor_write'])

    def test_normal_updates_nig_only_or_negative_memory_on_fp(self):
        r=np.ones(len(e4.CORE4))/len(e4.CORE4)
        b0=self.state(False);b1=self.state(True)
        patches=np.tile(self.x,(8,1))
        for st in (b0,b1):
            e4.apply_verified_feedback(st,is_defect=False,is_false_alarm=True,category='product',
                vector_raw=self.x,top8_patches=patches,source=None,responsibility=None,
                history=e4.SourceCentroids.empty(),permutation=self.perm,eta=.5)
        self.assertEqual(b0.fingerprint_nig('product'),b1.fingerprint_nig('product'))
        self.assertEqual(len(b0.neg_banks),0)
        self.assertEqual(b1.neg_banks['product'].observations,8)
        for g in e4.CORE4:
            np.testing.assert_array_equal(b0.router_memory[g]['sum'],b1.router_memory[g]['sum'])

    def test_credit_scorer_keeps_fractional_effective_count(self):
        b=copy.deepcopy(self.mem)
        base=e4.score_routing_fractional(self.x,b,self.prior)
        b[self.source]['sum']=self.x*.25
        b[self.source]['n']+=.25
        out=e4.score_routing_fractional(self.x,b,self.prior)
        self.assertNotEqual(out[0],base[0])
        self.assertTrue(np.isfinite(out).all())

    def test_candidate_pool_reranking_and_empty_bank(self):
        features=np.zeros((12,3)); features[:,0]=1
        features[:4]=[1,0,0];features[4:]=[0,1,0]
        scores=np.linspace(.5,.4,12)
        bare,top=e4.corrected_evidence(features,scores,.1,None)
        np.testing.assert_array_equal(top,np.arange(8))
        bank=e4.PrototypeBank(capacity=8);bank.add(np.array([1.,0.,0.]))
        vec,reranked=e4.corrected_evidence(features,scores,.3,bank)
        self.assertFalse(np.array_equal(top,reranked))
        self.assertLess(vec[0],bare[0])

    def test_fixed_query_and_sentinel_separation(self):
        items=[]
        for c in ('p1','p2'):
            for src in (None,self.source):
                for i in range(10):
                    items.append({'category':c,'role':'test_normal' if src is None else 'defect',
                                  'image_id':f'{c}/{src}/{i}','defect_source_offline_only':src})
        a,b=e4.split_stream_sentinel(items,123)
        self.assertEqual(len(set(a)&set(b)),0)
        q=e4.uniform_query_positions(len(a),8,123)
        self.assertTrue(set(q).issubset(set(range(len(a)))))
        self.assertEqual(len(q),8)
        a2,b2=e4.split_stream_sentinel(items,123)
        self.assertEqual(a,a2);self.assertEqual(b,b2)

    def test_all_branch_sets(self):
        self.assertEqual(set(e4.branches_for('e4a','n0')),{'N0','N1'})
        self.assertEqual(set(e4.branches_for('e4b','n0')),{'D0','D1','D2','D3'})
        self.assertEqual(set(e4.branches_for('e4c','n0')),{'C00','C10','C01','C11'})


if __name__=='__main__':
    unittest.main()
