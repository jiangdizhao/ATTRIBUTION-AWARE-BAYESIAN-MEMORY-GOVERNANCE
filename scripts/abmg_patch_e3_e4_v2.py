#!/usr/bin/env python3
"""Source-CV patch-level E3/E4A mechanism validation (not deployment stress).

Input: E4 frozen-Sensor evidence with --candidates 784 (full 28x28 patch grid).
Only source categories are used. Test-category labels are revealed for offline
selection of verified-normal queries and evaluation; sentinel items never write.
Factor groups are fixed raw-coordinate partitions, NOT discovered causal factors.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln
from sklearn.metrics import roc_auc_score

CHECKPOINTS = (0, 4, 8, 16, 32)
POLICIES = ('frozen', 'broadcast', 'selective', 'shuffled')
MODELS = ('gaussian', 'nig')


def load(path, fold):
    try:
        a = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        a = torch.load(path, map_location='cpu')
    if a.get('schema') != 'abmg.e4.evidence.v1' or a.get('outer_fold') != fold:
        raise ValueError('Expected E4 source-only evidence from requested outer fold')
    patches = a['candidate_features']
    if patches.ndim != 3 or patches.shape[1] != 784 or len(a['items']) != len(patches):
        raise ValueError('Patch E3/E4 requires ALL 784 patch descriptors; re-extract E4 with --candidates 784')
    if patches.shape[2] % 8:
        raise ValueError('Feature dimension must be divisible by eight fixed groups')
    return a


def rank_samples(patches, n=32):
    """Equal-rank-spaced patches across the frozen full-image patch ordering."""
    idx = np.linspace(0, len(patches)-1, min(n, len(patches)), dtype=int)
    return np.asarray(patches[idx], dtype=np.float64)


class PatchState:
    """Eight independent diagonal normal beliefs; one effective sample per image.

    NIG has a Student-t posterior predictive; Gaussian uses point estimates with
    shrinkage toward the same development variance. No trainable encoder.
    """
    def __init__(self, mean, var, model='nig', groups=8):
        if model not in MODELS or mean.shape != var.shape:
            raise ValueError('bad model or prior')
        if mean.size % groups:
            raise ValueError('invalid group count')
        self.model, self.groups = model, groups
        self.mu = np.array(mean, dtype=float, copy=True)
        self.var0 = np.maximum(np.asarray(var, dtype=float), 1e-7)
        self.kappa = np.full(mean.size, 0.05, dtype=float)
        self.alpha = np.full(mean.size, 2.5, dtype=float)
        self.beta = 1.5 * self.var0.copy()
        self.mass = np.zeros(groups, dtype=float)
        self.m2 = np.zeros(mean.size, dtype=float)

    def update(self, patches, credit):
        p = np.asarray(patches, dtype=float)
        cr = np.asarray(credit, dtype=float)
        if p.ndim != 2 or p.shape[1] != self.mu.size or len(cr) != self.groups:
            raise ValueError('invalid patch or credit shape')
        if np.any(cr < 0) or np.any(~np.isfinite(cr)) or cr.sum() < 0:
            raise ValueError('invalid credit')
        for g, (ids, w) in enumerate(zip(np.array_split(np.arange(self.mu.size), self.groups), cr)):
            if w == 0: continue
            # Batch sufficient-statistic fractional Bayesian update. The whole
            # image contributes w effective observations, not w * n_patches.
            x = p[:, ids]
            xmean = x.mean(axis=0)
            internal = np.mean((x-xmean)**2, axis=0)
            k0 = self.kappa[ids].copy()
            m0 = self.mu[ids].copy()
            kn = k0 + w
            self.mu[ids] = (k0*m0+w*xmean)/kn
            self.kappa[ids] = kn
            self.beta[ids] += 0.5*w*internal + 0.5*(k0*w/kn)*(xmean-m0)**2
            self.alpha[ids] += 0.5*w
            self.mass[g] += w
            self.m2[ids] += w*internal + (k0*w/kn)*(xmean-m0)**2

    def surprise(self, patches):
        p = np.asarray(patches, dtype=float)
        if p.ndim != 2 or p.shape[1] != self.mu.size:
            raise ValueError('invalid patch shape')
        out = []
        for group, ids in enumerate(np.array_split(np.arange(self.mu.size), self.groups)):
            diff = p[:, ids]-self.mu[ids]
            if self.model == 'nig':
                nu = 2*self.alpha[ids]
                scale2 = np.maximum(self.beta[ids]*(self.kappa[ids]+1)/(self.alpha[ids]*self.kappa[ids]), 1e-9)
                ll = (gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(nu*np.pi*scale2)
                      -(nu+1)/2*np.log1p(diff**2/(nu*scale2)))
            else:
                # Plug-in Gaussian, regularized running variance estimate.
                v = np.maximum((self.var0[ids] + self.m2[ids])/(1.0+self.mass[group]), self.var0[ids]*0.05)
                ll = -0.5*(np.log(2*np.pi*v)+diff**2/v)
            out.append(-ll.mean(axis=1))
        return np.stack(out, axis=1)  # [patch, factor]

    def image_evidence(self, patches, k=8):
        surprise = self.surprise(patches)
        kk = min(k, len(surprise))
        return np.partition(surprise, -kk, axis=0)[-kk:].mean(axis=0)


def image_score(state, patches):
    return float(np.max(state.image_evidence(patches)))


def choose_credit(state, patches, policy, permutation):
    g = state.groups
    if policy == 'frozen': return np.zeros(g)
    if policy == 'broadcast': return np.full(g, 1.0/g)
    # Responsibility is PRE-UPDATE surprise, not a known true factor address.
    evidence = state.image_evidence(patches)
    j = int(np.argmax(evidence))
    if policy == 'shuffled': j = int(permutation[j])
    if policy not in POLICIES: raise ValueError(policy)
    out = np.zeros(g)
    out[j] = 1.0
    return out


def evaluate(state, normal, defects):
    normal_scores = [image_score(state, x) for x in normal]
    defect_scores = [image_score(state, x) for x in defects]
    auc = (float(roc_auc_score([0]*len(normal_scores)+[1]*len(defect_scores),
                              normal_scores+defect_scores))
           if normal_scores and defect_scores else None)
    return {'normal_surprise': float(np.mean(normal_scores)) if normal_scores else None,
            'defect_surprise': float(np.mean(defect_scores)) if defect_scores else None,
            'auroc': auc, 'n_normal': len(normal_scores), 'n_defect': len(defect_scores)}


def prior_from_dev(a, heldout, sample_images=3, sample_patches=16):
    """Source-CV development TRAIN normals only; no heldout/sentinel access."""
    rows = []
    seen = {}
    for i, item in enumerate(a['items']):
        cat = item['category']
        if cat in heldout or item['role'] != 'train_normal' or seen.get(cat, 0) >= sample_images:
            continue
        seen[cat] = seen.get(cat, 0)+1
        rows.append(rank_samples(a['candidate_features'][i].numpy(), sample_patches))
    if not rows: raise ValueError('No CV development training normals available')
    x = np.concatenate(rows, axis=0)
    mean = np.mean(x, axis=0)
    var = np.maximum(np.var(x, axis=0), 1e-5)
    return mean, var


def run(a, mode, cv_fold, budget, seed, support=2, patch_batch=32):
    if mode not in ('e3','e4a'): raise ValueError('stage only e3 or e4a')
    heldout = set(a['source_cv_groups'][cv_fold])
    mean, var = prior_from_dev(a, heldout)
    bycat = {c: {'train_normal': [], 'test_normal': [], 'defect': []} for c in heldout}
    for i, row in enumerate(a['items']):
        if row['category'] in bycat and row['role'] in bycat[row['category']]:
            bycat[row['category']][row['role']].append(i)
    rng = np.random.default_rng(seed+cv_fold*1009)
    permutation = np.roll(np.arange(8), 1)  # fixed derangement, never identity
    out = {'schema':'abmg.patch_e3_e4.v2', 'mode':mode, 'outer_fold':a['outer_fold'],
           'source_cv_fold':cv_fold, 'categories':sorted(heldout), 'budget':budget,
           'support':support, 'query_policy':'offline verified-normal high-surprise selection; NOT deployable',
           'feature_scope':'ALL 784 frozen-Sensor raw patches, raw-coordinate 8 groups',
           'factor_semantics':'operational fixed coordinate groups; not validated disentangled/causal factors',
           'results':[]}
    for cat in sorted(heldout):
        ids = bycat[cat]
        train = rng.permutation(ids['train_normal']).tolist()
        sent_normal = sorted(ids['test_normal'])
        sent_defect = sorted(ids['defect'])
        if len(train) < support+1 or not sent_normal or not sent_defect:
            out.setdefault('skipped',[]).append({'category':cat,'train_normal':len(train),
                                                 'sent_normal':len(sent_normal),'sent_defect':len(sent_defect)})
            continue
        patches = a['candidate_features']
        supports = train[:support]
        candidate = train[support:]
        sentinel_n = [patches[i][:64].numpy().astype('float64') for i in sent_normal]
        sentinel_d = [patches[i][:64].numpy().astype('float64') for i in sent_defect]
        types = MODELS if mode == 'e3' else ('nig',)
        branches = {}
        for model in types:
            state = PatchState(mean, var, model)
            for idx in supports:
                state.update(rank_samples(patches[idx].numpy(), patch_batch), np.ones(8))
            branches[model] = state
        # Selection uses one frozen initial NIG reference; no queried sentinel labels.
        selection_state = branches['nig'] if 'nig' in branches else PatchState(mean,var,'nig')
        if 'nig' not in branches:
            for idx in supports: selection_state.update(rank_samples(patches[idx].numpy(), patch_batch), np.ones(8))
        ordered = sorted(candidate, key=lambda i: image_score(selection_state, patches[i][:64].numpy()), reverse=True)
        queried = ordered[:budget]
        states = ({m: branches[m] for m in types} if mode == 'e3' else
                  {p: copy.deepcopy(branches['nig']) for p in POLICIES})
        checkpoints = []
        for t in range(len(queried)+1):
            if t in CHECKPOINTS or t == len(queried):
                metrics = {name:{**evaluate(st,sentinel_n,sentinel_d),
                                 'effective_mass':st.mass.tolist()}
                           for name,st in states.items()}
                checkpoints.append({'queries':t,'branches':metrics})
            if t == len(queried): break
            idx = queried[t]
            z = rank_samples(patches[idx].numpy(), patch_batch)
            for name, st in states.items():
                credit = np.ones(8) if mode=='e3' else choose_credit(st,z,name,permutation)
                if credit.any(): st.update(z,credit)
        out['results'].append({'category':cat,'n_support':len(supports),
                               'n_queries':len(queried), 'n_sentinel_normal':len(sent_normal),
                               'n_sentinel_defect':len(sent_defect), 'checkpoints':checkpoints})
    if not out['results']: raise RuntimeError('No evaluable categories; increase normal/defect extraction caps')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['e3','e4a'],required=True)
    p.add_argument('--evidence',required=True)
    p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--source-cv-fold',type=int,choices=range(4),default=0)
    p.add_argument('--budget',type=int,default=16)
    p.add_argument('--support',type=int,default=2)
    p.add_argument('--patch-batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out-dir',default='outputs/patch_e3_e4_v2')
    args=p.parse_args()
    if args.budget < 1 or args.support < 1 or args.patch_batch < 1: p.error('positive budget/support/patch-batch required')
    artifact=load(args.evidence,args.fold)
    result=run(artifact,args.mode,args.source_cv_fold,args.budget,args.seed,args.support,args.patch_batch)
    dest=Path(args.out_dir)/f'fold_{args.fold}'/f'cv_{args.source_cv_fold}'
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'patch_{args.mode}_audit.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Wrote {path} ({len(result["results"])} product categories)')


if __name__=='__main__': main()
