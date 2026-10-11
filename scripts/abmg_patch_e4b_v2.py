#!/usr/bin/env python3
"""E4B: source-enriched patch-factor responsibility memory, source-only audit.

Controlled, informative source-verified feedback on held-out source-CV products.
This tests q_d enrichment in factor responsibility space, NOT defect correction
of image anomaly scores. No outer-target category or sentinel is used to write.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

try:
    from abmg_patch_e3_e4_v2 import (CHECKPOINTS, PatchState, choose_credit, load,
                                      prior_from_dev, rank_samples)
except ImportError:
    from scripts.abmg_patch_e3_e4_v2 import (CHECKPOINTS, PatchState, choose_credit,
                                              load, prior_from_dev, rank_samples)

CORE4 = ('AK', 'HS', 'QS', 'ZW')
POLICIES = ('frozen', 'broadcast', 'correct_source', 'shuffled_source')


class SourceMemory:
    def __init__(self, starts):
        self.n = {d:1.0 for d in CORE4}
        self.total = {d:np.asarray(starts[d],dtype=np.float64).copy() for d in CORE4}

    def update(self, r, source, policy, wrong):
        if source not in CORE4: raise ValueError('source not Core4')
        if policy == 'frozen': return
        credits = {d:0.0 for d in CORE4}
        if policy == 'broadcast': credits = {d:0.25 for d in CORE4}
        elif policy == 'correct_source': credits[source] = 1.0
        elif policy == 'shuffled_source': credits[wrong[source]] = 1.0
        else: raise ValueError(policy)
        for d,w in credits.items():
            self.total[d] += w*np.asarray(r,dtype=np.float64)
            self.n[d] += w

    def predict(self,r):
        rr=np.asarray(r,dtype=float)
        rr=rr/max(np.linalg.norm(rr),1e-12)
        logits=[]
        for d in CORE4:
            q=self.total[d]/self.n[d]
            logits.append(float(rr@q/max(np.linalg.norm(q),1e-12)))
        return CORE4[int(np.argmax(logits))], np.asarray(logits)


def responsibility(state, patch_tensor):
    return choose_credit(state, np.asarray(patch_tensor[:64], dtype=np.float64),
                         'selective', np.roll(np.arange(8),1))


def normal_state(a,cat,bycat,prior,support=2,batch=32):
    mean,var,calibration=prior
    state=PatchState(mean,var,'nig',calibration=calibration['nig'])
    normal=bycat[cat]['train_normal']
    if len(normal)<support: raise ValueError('product lacks training normals: '+cat)
    for i in sorted(normal)[:support]:
        state.update(rank_samples(a['candidate_features'][i].numpy(),batch),np.ones(8))
    return state


def evaluate(mem, observations):
    ytrue=[]; ypred=[]; margin=[]
    for label,r in observations:
        pred,logits=mem.predict(r)
        ytrue.append(label); ypred.append(pred)
        true=CORE4.index(label)
        margin.append(float(logits[true]-np.max(np.delete(logits,true))))
    return {'macro_f1':float(f1_score(ytrue,ypred,labels=list(CORE4),average='macro',zero_division=0)),
            'accuracy':float(np.mean(np.asarray(ytrue)==np.asarray(ypred))),
            'true_source_margin':float(np.mean(margin)), 'n_sentinel':len(observations)}


def run(a,cv_fold,budget=32,seed=0,support=2):
    held=set(a['source_cv_groups'][cv_fold])
    dev=set(a['source_categories'])-held
    prior=prior_from_dev(a,held)
    bycat=defaultdict(lambda:defaultdict(list))
    for i,item in enumerate(a['items']):
        bycat[item['category']][item['role']].append(i)
    # Build dev-product normal states with no defect labels; one offline source
    # exemplar for each known source is the shared predeployment anchor.
    state_cache={}
    def state(cat):
        if cat not in state_cache: state_cache[cat]=normal_state(a,cat,bycat,prior,support)
        return state_cache[cat]
    dev_samples={d:[] for d in CORE4}
    for i,item in enumerate(a['items']):
        d=item['defect_source_offline_only']
        if item['category'] in dev and item['role']=='defect' and d in CORE4:
            dev_samples[d].append((item['category'],i))
    missing=[d for d in CORE4 if not dev_samples[d]]
    if missing: raise ValueError('Source-CV development missing source anchors: '+str(missing))
    starters={}
    for d in CORE4:
        cat,i=sorted(dev_samples[d])[0]
        starters[d]=responsibility(state(cat),a['candidate_features'][i].numpy())
    rng=np.random.default_rng(seed+cv_fold*1009)
    test_rows=defaultdict(list)
    for i,item in enumerate(a['items']):
        d=item['defect_source_offline_only']
        if item['category'] in held and item['role']=='defect' and d in CORE4:
            test_rows[(item['category'],d)].append(i)
    query_pool=[]; sentinel=[]
    for (cat,d),ids in sorted(test_rows.items()):
        ids=rng.permutation(ids).tolist()
        if len(ids)>1:
            ns=max(1,int(round(.35*len(ids))))
            ns=min(ns,len(ids)-1)
            sentinel.extend(ids[:ns]);query_pool.extend(ids[ns:])
        else: sentinel.extend(ids)
    if not query_pool or not sentinel: raise RuntimeError('Insufficient source-CV feedback/sentinel samples')
    # Offline source-balanced schedule; same fixed examples in every branch.
    bysource=defaultdict(list)
    for i in query_pool: bysource[a['items'][i]['defect_source_offline_only']].append(i)
    for rows in bysource.values(): rng.shuffle(rows)
    queries=[]
    while len(queries)<budget and any(bysource.values()):
        for d in CORE4:
            if bysource[d] and len(queries)<budget: queries.append(bysource[d].pop())
    frozen_responsibilities={i:responsibility(state(a['items'][i]['category']),a['candidate_features'][i].numpy())
                             for i in set(sentinel+queries)}
    sentinel_obs=[(a['items'][i]['defect_source_offline_only'],frozen_responsibilities[i]) for i in sentinel]
    wrong={'AK':'HS','HS':'AK','QS':'ZW','ZW':'QS'}
    memory={p:SourceMemory(starters) for p in POLICIES}
    cps=[]
    for t in range(len(queries)+1):
        if t in CHECKPOINTS or t == len(queries):
            cps.append({'queries':t,'branches':{p:evaluate(mem,sentinel_obs) for p,mem in memory.items()}})
        if t==len(queries): break
        idx=queries[t]
        d=a['items'][idx]['defect_source_offline_only']
        r=frozen_responsibilities[idx]
        for p in POLICIES:memory[p].update(r,d,p,wrong)
    return {'schema':'abmg.patch_e4b_source_attribution.v2','outer_fold':a['outer_fold'],
            'source_cv_fold':cv_fold,'query_budget':budget,'queries_realized':len(queries),
            'n_sentinel':len(sentinel),'source_labels':list(CORE4),
            'readout':'future source-routing only; anomaly score unaffected',
            'query_policy':'offline source-balanced verified defect labels, not deployment',
            'initial_source_memory':'one verified development example per source',
            'controls':list(POLICIES),'sentinel_image_ids':[a['items'][i]['image_id'] for i in sentinel],
            'query_image_ids':[a['items'][i]['image_id'] for i in queries], 'checkpoints':cps}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',required=True,type=int,choices=range(5))
    p.add_argument('--evidence',required=True)
    p.add_argument('--source-cv-fold',default=0,type=int,choices=range(4))
    p.add_argument('--budget',default=32,type=int)
    p.add_argument('--support',default=2,type=int)
    p.add_argument('--seed',default=0,type=int)
    p.add_argument('--out-dir',default='outputs/patch_e3_e4_v2')
    arg=p.parse_args()
    if arg.budget<1 or arg.support<1: p.error('budget/support must be positive')
    a=load(arg.evidence,arg.fold)
    out=run(a,arg.source_cv_fold,arg.budget,arg.seed,arg.support)
    dest=Path(arg.out_dir)/f'fold_{arg.fold}'/f'cv_{arg.source_cv_fold}'
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'patch_e4b_audit.json'
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Wrote {path}: {out["queries_realized"]} verified source queries')


if __name__=='__main__':main()
